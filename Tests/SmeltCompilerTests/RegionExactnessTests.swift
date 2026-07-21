// RegionExactnessTests — Per-region exactness contracts for kernel work.
//
// A "region" is a kernel or producer-consumer chain (e.g., rms_norm →
// affine_matvec, apply_rope, kv_cache_update, attention_decode) that
// the optimizer might fuse, schedule, or replace. Each test in this
// file pins down what "correct" means for one region:
//
//   1. Build the region's inputs synthetically — quantized weights for
//      matvec regions (`makeQuantizedAffineFixture`), FP16 vectors for
//      attention regions (`makeFloat16Vector`).
//   2. Compute the expected output on CPU. Stage in FP32 inside, round
//      to FP16 at the same boundaries the kernel rounds at.
//   3. Run the GPU path (one or more kernel dispatches). Assert it
//      matches the CPU reference within the region's tolerance.
//   4. (Future, when a fused kernel ships:) run the fused candidate and
//      assert it matches the GPU staged output byte-for-byte.
//
// To add a region test: copy one of the test methods, then write:
//   - make<Region>Inputs(...) — synthetic inputs
//   - reference<Region>CPU(...) — the math, FP32 acc, FP16 boundaries
//     (call `referenceAffineMatvecRow` for matvec-tailed regions)
//   - run<Region>GPU(...) — Metal kernel dispatches in order
//   - one or more shapes — cover edge cases per the kernel's contract:
//     tail rows for matvec, layout variants for rope, threadgroup
//     widths that exercise full reduction trees for attention, etc.

import Foundation
import XCTest
import Metal

final class RegionExactnessTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        device = dev
        queue = dev.makeCommandQueue()
    }

    // MARK: - Threadgroup contracts (named to match kernel comments)

    private static let matvecRowsPerThreadgroup = 8
    private static let matvecThreadsPerThreadgroup = 64

    /// Dispatch contract for the affine_matvec / fused_affine_gate_up_swiglu /
    /// fused_dual_affine_matvec kernel family: ceil(R / ROWS_PER_TG)
    /// threadgroups × THREADS_PER_TG threads. (See `lut_matvec.metal:1027`.)
    private func dispatchAffineMatvec(encoder: MTLComputeCommandEncoder, rows: Int) {
        let rowsPerTG = Self.matvecRowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.matvecThreadsPerThreadgroup, height: 1, depth: 1
            )
        )
    }

    // MARK: - Tolerance budgets per region

    private static let tripwireMinAbs: Float = 1e-2
    /// Linear matvec math (single or dual) — FP32 accumulation, FP16 rounding
    /// at the output. At the fixture's ~52 magnitude, one Float16 ULP is
    /// 0.03125; keep the budget just above that representable step.
    private static let affineMatvecTolerance: Float = 4e-2
    /// silu(gate)*up amplifies ULP differences in gate by the up factor;
    /// gets its own slacker budget. At the fixture's ~306 magnitude, one
    /// Float16 ULP is 0.25.
    private static let gateUpSwigluTolerance: Float = 3e-1

    // MARK: - rms_norm_1pw → affine_matvec
    //
    // Layer-norm into a quantized matvec is the dominant decode pattern in
    // Qwen / Llama. The fused candidate (`fused_rms_norm_affine_matvec`,
    // catalog case 44) is declared in the catalog but not yet emitted; this
    // test pins down the staged reference now so any future fused
    // implementation has a contract to meet.

    func testRmsNormAffineMatvec_StagedMatchesReference_AlignedShape() throws {
        try runRmsAffineRegion(cols: 128, rows: 256, groupSize: 64)
    }

    /// rows=257 forces the kernel's tail-row handling (one threadgroup
    /// processes a partial row count). Bugs in `validRows`/`sgValidRows`
    /// guard logic only surface when rows is not a multiple of ROWS_PER_TG.
    func testRmsNormAffineMatvec_StagedMatchesReference_TailShape() throws {
        try runRmsAffineRegion(cols: 128, rows: 257, groupSize: 64)
    }

    private func runRmsAffineRegion(cols: Int, rows: Int, groupSize: Int) throws {
        let eps: Float = 1e-6

        let normPipeline = makeComputePipeline(
            device: device, shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile rms_norm_1pw pipeline")

        let matvecPipeline = makeComputePipeline(
            device: device, shaderFile: "lut_matvec.metal", functionName: "affine_matvec",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(matvecPipeline == nil, "Could not compile affine_matvec pipeline")

        let inputs = makeRmsAffineInputs(
            cols: cols, rows: rows, groupSize: groupSize, seed: 1
        )

        let cpuExpected = referenceRmsAffineCPU(
            inputs: inputs, rows: rows, cols: cols, groupSize: groupSize, eps: eps
        )

        let gpuActual = try runStagedRmsAffineGPU(
            normPipeline: normPipeline!, matvecPipeline: matvecPipeline!,
            inputs: inputs, rows: rows, cols: cols, eps: eps
        )

        // Trip-wire: a wiring bug that wrote zeros to both buffers would
        // otherwise compare-equal to a CPU reference that read zeros.
        // Output magnitudes for these inputs are O(1); 1e-2 catches "all
        // zero" without false-failing on small numerical drift.
        let gpuMaxAbs = gpuActual.map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(gpuMaxAbs, Self.tripwireMinAbs, "GPU output is suspiciously close to zero")

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "rms_norm + affine_matvec staged",
            shape: "[c\(cols),r\(rows),g\(groupSize)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        // FP32 accumulation in both paths plus matching group/row layout
        // gives byte-equal output for typical shapes; a small tolerance
        // absorbs ULP variation when reduction order diverges (e.g.,
        // wider kernels with simd_sum across more lanes than groups).
        XCTAssertLessThan(maxDiff, Self.affineMatvecTolerance, "Staged GPU diverged from CPU reference (rows=\(rows))")
    }

    private struct RmsAffineInputs {
        let input: [Float16]
        let normWeight: [Float16]
        let weights: [UInt8]
        let scales: [Float16]
        let biases: [Float16]
    }

    private func makeRmsAffineInputs(
        cols: Int,
        rows: Int,
        groupSize: Int,
        seed: Int
    ) -> RmsAffineInputs {
        let (input, normWeight) = makeRmsNormInputs(cols: cols, seed: seed)
        let (weights, scales, biases) = makeQuantizedAffineFixture(
            rows: rows, cols: cols, groupSize: groupSize, seed: seed
        )
        return RmsAffineInputs(
            input: input, normWeight: normWeight,
            weights: weights, scales: scales, biases: biases
        )
    }

    private func referenceRmsAffineCPU(
        inputs: RmsAffineInputs,
        rows: Int,
        cols: Int,
        groupSize: Int,
        eps: Float
    ) -> [Float] {
        let normalized = cpuRmsNorm1pw(
            input: inputs.input, normWeight: inputs.normWeight,
            cols: cols, eps: eps
        )
        var output = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            output[r] = referenceAffineMatvecRow(
                weights: inputs.weights, scales: inputs.scales, biases: inputs.biases,
                input: normalized, row: r, cols: cols, groupSize: groupSize
            )
        }
        return output
    }

    private func runStagedRmsAffineGPU(
        normPipeline: MTLComputePipelineState,
        matvecPipeline: MTLComputePipelineState,
        inputs: RmsAffineInputs,
        rows: Int,
        cols: Int,
        eps: Float
    ) throws -> [Float16] {
        let inputBuf = try makeSharedBuffer(device: device, inputs.input)
        let normWeightBuf = try makeSharedBuffer(device: device, inputs.normWeight)
        let normOutputBuf = try makeSharedBuffer(device: device, count: cols, of: Float16.self)
        let weightsBuf = try makeSharedBuffer(device: device, inputs.weights)
        let scalesBuf = try makeSharedBuffer(device: device, inputs.scales)
        let biasesBuf = try makeSharedBuffer(device: device, inputs.biases)
        let outputBuf = try makeSharedBuffer(device: device, count: rows, of: Float16.self)

        try runOnGPU(queue: queue) { enc in
            encodeRmsNorm1pw(
                encoder: enc, pipeline: normPipeline,
                input: inputBuf, normWeight: normWeightBuf, output: normOutputBuf,
                cols: cols, eps: eps
            )

            enc.setComputePipelineState(matvecPipeline)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(normOutputBuf, offset: 0, index: 3)
            enc.setBuffer(outputBuf, offset: 0, index: 4)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 5)
            dispatchAffineMatvec(encoder: enc, rows: rows)
        }

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: outPtr, count: rows))
    }

    // MARK: - rms_norm_1pw → fused_affine_gate_up_swiglu
    //
    // SwiGLU FFN's first half: input → norm → (gate matvec, up matvec) →
    // silu(gate) * up. The fused candidate (`fused_affine_gate_up_swiglu`,
    // catalog case 42, kernel exists at lut_matvec.metal:1903) collapses
    // the dual matvec + activation into one kernel; the staged reference
    // is rms_norm + two affine_matvec dispatches + a silu*up step. We
    // run the fused kernel as the GPU candidate and check it against
    // the all-CPU staged reference.

    func testRmsNormGateUpSwiglu_FusedMatchesReference_AlignedShape() throws {
        try runRmsGateUpSwigluRegion(cols: 128, rows: 256, groupSize: 64)
    }

    /// Tail-row coverage matching the affine_matvec test.
    func testRmsNormGateUpSwiglu_FusedMatchesReference_TailShape() throws {
        try runRmsGateUpSwigluRegion(cols: 128, rows: 257, groupSize: 64)
    }

    private func runRmsGateUpSwigluRegion(cols: Int, rows: Int, groupSize: Int) throws {
        let eps: Float = 1e-6

        let normPipeline = makeComputePipeline(
            device: device, shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile rms_norm_1pw pipeline")

        let fusedPipeline = makeComputePipeline(
            device: device, shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(fusedPipeline == nil, "Could not compile fused_affine_gate_up_swiglu pipeline")

        let inputs = makeRmsGateUpSwigluInputs(
            cols: cols, rows: rows, groupSize: groupSize, seed: 1
        )

        let cpuExpected = referenceRmsGateUpSwigluCPU(
            inputs: inputs, rows: rows, cols: cols, groupSize: groupSize, eps: eps
        )

        let gpuActual = try runFusedRmsGateUpSwigluGPU(
            normPipeline: normPipeline!, fusedPipeline: fusedPipeline!,
            inputs: inputs, rows: rows, cols: cols, eps: eps
        )

        let gpuMaxAbs = gpuActual.map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(gpuMaxAbs, Self.tripwireMinAbs, "GPU output is suspiciously close to zero")

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "rms_norm + fused_affine_gate_up_swiglu",
            shape: "[c\(cols),r\(rows),g\(groupSize)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        // silu(x) * y is more sensitive than a linear matvec — small ULP
        // differences in gate get amplified by the up factor. 1e-2 is
        // loose enough for FMA-reorder noise without missing real bugs.
        XCTAssertLessThan(maxDiff, Self.gateUpSwigluTolerance, "Fused GPU diverged from CPU reference (rows=\(rows))")
    }

    private struct RmsGateUpSwigluInputs {
        let input: [Float16]
        let normWeight: [Float16]
        let gateWeights: [UInt8]
        let gateScales: [Float16]
        let gateBiases: [Float16]
        let upWeights: [UInt8]
        let upScales: [Float16]
        let upBiases: [Float16]
    }

    private func makeRmsGateUpSwigluInputs(
        cols: Int,
        rows: Int,
        groupSize: Int,
        seed: Int
    ) -> RmsGateUpSwigluInputs {
        // Positive input bias keeps post-norm activations above silu's
        // FP16 underflow cliff: a pure mean-zero sine collapses gate
        // logits to bias_q*xsum ≈ 0, then silu(0)*up = 0, defeating the
        // trip-wire and the regression detection it backstops.
        let (input, normWeight) = makeRmsNormInputs(cols: cols, seed: seed, inputBias: 0.2)
        let gate = makeQuantizedAffineFixture(
            rows: rows, cols: cols, groupSize: groupSize, seed: seed
        )
        // Different seed for up so gate and up matrices are uncorrelated;
        // a bug that swaps gate/up indices would still pass an all-equal
        // weights check, so this is part of the test's coverage.
        let up = makeQuantizedAffineFixture(
            rows: rows, cols: cols, groupSize: groupSize, seed: seed + 1
        )
        return RmsGateUpSwigluInputs(
            input: input, normWeight: normWeight,
            gateWeights: gate.weights, gateScales: gate.scales, gateBiases: gate.biases,
            upWeights: up.weights, upScales: up.scales, upBiases: up.biases
        )
    }

    private func referenceRmsGateUpSwigluCPU(
        inputs: RmsGateUpSwigluInputs,
        rows: Int,
        cols: Int,
        groupSize: Int,
        eps: Float
    ) -> [Float] {
        let normalized = cpuRmsNorm1pw(
            input: inputs.input, normWeight: inputs.normWeight,
            cols: cols, eps: eps
        )
        var output = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            // Gate and up matvec results stay in FP32 — the fused kernel
            // applies silu * up before any FP16 round, so the reference
            // must match that ordering.
            let gateLogit = referenceAffineMatvecRow(
                weights: inputs.gateWeights, scales: inputs.gateScales,
                biases: inputs.gateBiases, input: normalized,
                row: r, cols: cols, groupSize: groupSize, roundToFP16: false
            )
            let upLogit = referenceAffineMatvecRow(
                weights: inputs.upWeights, scales: inputs.upScales,
                biases: inputs.upBiases, input: normalized,
                row: r, cols: cols, groupSize: groupSize, roundToFP16: false
            )
            let silu = gateLogit / (1 + exp(-gateLogit))
            output[r] = Float(Float16(silu * upLogit))
        }
        return output
    }

    private func runFusedRmsGateUpSwigluGPU(
        normPipeline: MTLComputePipelineState,
        fusedPipeline: MTLComputePipelineState,
        inputs: RmsGateUpSwigluInputs,
        rows: Int,
        cols: Int,
        eps: Float
    ) throws -> [Float16] {
        let inputBuf = try makeSharedBuffer(device: device, inputs.input)
        let normWeightBuf = try makeSharedBuffer(device: device, inputs.normWeight)
        let normOutputBuf = try makeSharedBuffer(device: device, count: cols, of: Float16.self)
        let gateWeightsBuf = try makeSharedBuffer(device: device, inputs.gateWeights)
        let gateScalesBuf = try makeSharedBuffer(device: device, inputs.gateScales)
        let gateBiasesBuf = try makeSharedBuffer(device: device, inputs.gateBiases)
        let upWeightsBuf = try makeSharedBuffer(device: device, inputs.upWeights)
        let upScalesBuf = try makeSharedBuffer(device: device, inputs.upScales)
        let upBiasesBuf = try makeSharedBuffer(device: device, inputs.upBiases)
        let outputBuf = try makeSharedBuffer(device: device, count: rows, of: Float16.self)

        try runOnGPU(queue: queue) { enc in
            encodeRmsNorm1pw(
                encoder: enc, pipeline: normPipeline,
                input: inputBuf, normWeight: normWeightBuf, output: normOutputBuf,
                cols: cols, eps: eps
            )

            enc.setComputePipelineState(fusedPipeline)
            enc.setBuffer(gateWeightsBuf, offset: 0, index: 0)
            enc.setBuffer(gateScalesBuf, offset: 0, index: 1)
            enc.setBuffer(gateBiasesBuf, offset: 0, index: 2)
            enc.setBuffer(upWeightsBuf, offset: 0, index: 3)
            enc.setBuffer(upScalesBuf, offset: 0, index: 4)
            enc.setBuffer(upBiasesBuf, offset: 0, index: 5)
            enc.setBuffer(normOutputBuf, offset: 0, index: 6)
            enc.setBuffer(outputBuf, offset: 0, index: 7)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 8)
            dispatchAffineMatvec(encoder: enc, rows: rows)
        }

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: outPtr, count: rows))
    }

    // MARK: - rms_norm_1pw → fused_dual_affine_matvec
    //
    // Production use case: K and V projections in GQA attention. Both
    // weight matrices have the same output shape and read the same
    // post-norm input, so the fused kernel
    // (`fused_dual_affine_matvec`, catalog case 41, kernel at
    // lut_matvec.metal:1694) loads the input once and runs two
    // matvecs in parallel. The exactness contract is byte-equal
    // against two independent affine_matvec computations on the
    // same staged input.

    func testRmsNormDualMatvec_FusedMatchesReference_AlignedShape() throws {
        try runRmsDualMatvecRegion(cols: 128, rows: 256, groupSize: 64)
    }

    /// Tail-row coverage matching the other regions.
    func testRmsNormDualMatvec_FusedMatchesReference_TailShape() throws {
        try runRmsDualMatvecRegion(cols: 128, rows: 257, groupSize: 64)
    }

    private func runRmsDualMatvecRegion(cols: Int, rows: Int, groupSize: Int) throws {
        let eps: Float = 1e-6

        let normPipeline = makeComputePipeline(
            device: device, shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile rms_norm_1pw pipeline")

        let dualPipeline = makeComputePipeline(
            device: device, shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(dualPipeline == nil, "Could not compile fused_dual_affine_matvec pipeline")

        let inputs = makeRmsDualMatvecInputs(
            cols: cols, rows: rows, groupSize: groupSize, seed: 1
        )

        let cpuExpected = referenceRmsDualMatvecCPU(
            inputs: inputs, rows: rows, cols: cols, groupSize: groupSize, eps: eps
        )

        let gpuActual = try runFusedRmsDualMatvecGPU(
            normPipeline: normPipeline!, dualPipeline: dualPipeline!,
            inputs: inputs, rows: rows, cols: cols, eps: eps
        )

        // Per-buffer trip-wire: a kernel that wrote correct output to
        // one buffer and zeros to the other would still clear a single
        // combined max-abs check. Assert each output is non-trivial
        // before the concatenated max-diff comparison.
        let kMaxAbs = gpuActual[0..<rows].map { abs(Float($0)) }.max() ?? 0
        let vMaxAbs = gpuActual[rows..<rows * 2].map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(kMaxAbs, Self.tripwireMinAbs, "K output is suspiciously close to zero")
        XCTAssertGreaterThan(vMaxAbs, Self.tripwireMinAbs, "V output is suspiciously close to zero")
        let gpuMaxAbs = max(kMaxAbs, vMaxAbs)

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "rms_norm + fused_dual_affine_matvec",
            shape: "[c\(cols),r\(rows),g\(groupSize)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(maxDiff, Self.affineMatvecTolerance, "Fused dual GPU diverged from CPU reference (rows=\(rows))")
    }

    private struct RmsDualMatvecInputs {
        let input: [Float16]
        let normWeight: [Float16]
        // Production names: K and V projections in GQA attention. Both
        // matrices have the same output shape and read the same input.
        let kWeights: [UInt8]
        let kScales: [Float16]
        let kBiases: [Float16]
        let vWeights: [UInt8]
        let vScales: [Float16]
        let vBiases: [Float16]
    }

    private func makeRmsDualMatvecInputs(
        cols: Int,
        rows: Int,
        groupSize: Int,
        seed: Int
    ) -> RmsDualMatvecInputs {
        let (input, normWeight) = makeRmsNormInputs(cols: cols, seed: seed)
        let k = makeQuantizedAffineFixture(
            rows: rows, cols: cols, groupSize: groupSize, seed: seed
        )
        // Different seed for V so a bug that swaps output1/output2
        // bindings can't compare-equal — same defense the gate_up
        // region uses for gate vs up. Note: with seed=1, K fixtures
        // are byte-identical to gate_weights in the gate_up region
        // (seed=1) and V fixtures match up_weights (seed=2). A
        // weight-fixture-generator regression would surface in both
        // tests at once, which is the intended cross-region signal.
        let v = makeQuantizedAffineFixture(
            rows: rows, cols: cols, groupSize: groupSize, seed: seed + 1
        )
        return RmsDualMatvecInputs(
            input: input, normWeight: normWeight,
            kWeights: k.weights, kScales: k.scales, kBiases: k.biases,
            vWeights: v.weights, vScales: v.scales, vBiases: v.biases
        )
    }

    private func referenceRmsDualMatvecCPU(
        inputs: RmsDualMatvecInputs,
        rows: Int,
        cols: Int,
        groupSize: Int,
        eps: Float
    ) -> [Float] {
        let normalized = cpuRmsNorm1pw(
            input: inputs.input, normWeight: inputs.normWeight,
            cols: cols, eps: eps
        )
        // Block layout: K outputs followed by V outputs, so the GPU
        // readback (also concatenated) lines up element-wise.
        var output = [Float](repeating: 0, count: rows * 2)
        for r in 0..<rows {
            output[r] = referenceAffineMatvecRow(
                weights: inputs.kWeights, scales: inputs.kScales, biases: inputs.kBiases,
                input: normalized, row: r, cols: cols, groupSize: groupSize
            )
            output[rows + r] = referenceAffineMatvecRow(
                weights: inputs.vWeights, scales: inputs.vScales, biases: inputs.vBiases,
                input: normalized, row: r, cols: cols, groupSize: groupSize
            )
        }
        return output
    }

    private func runFusedRmsDualMatvecGPU(
        normPipeline: MTLComputePipelineState,
        dualPipeline: MTLComputePipelineState,
        inputs: RmsDualMatvecInputs,
        rows: Int,
        cols: Int,
        eps: Float
    ) throws -> [Float16] {
        let inputBuf = try makeSharedBuffer(device: device, inputs.input)
        let normWeightBuf = try makeSharedBuffer(device: device, inputs.normWeight)
        let normOutputBuf = try makeSharedBuffer(device: device, count: cols, of: Float16.self)
        let kWeightsBuf = try makeSharedBuffer(device: device, inputs.kWeights)
        let kScalesBuf = try makeSharedBuffer(device: device, inputs.kScales)
        let kBiasesBuf = try makeSharedBuffer(device: device, inputs.kBiases)
        let vWeightsBuf = try makeSharedBuffer(device: device, inputs.vWeights)
        let vScalesBuf = try makeSharedBuffer(device: device, inputs.vScales)
        let vBiasesBuf = try makeSharedBuffer(device: device, inputs.vBiases)
        let kOutputBuf = try makeSharedBuffer(device: device, count: rows, of: Float16.self)
        let vOutputBuf = try makeSharedBuffer(device: device, count: rows, of: Float16.self)

        try runOnGPU(queue: queue) { enc in
            encodeRmsNorm1pw(
                encoder: enc, pipeline: normPipeline,
                input: inputBuf, normWeight: normWeightBuf, output: normOutputBuf,
                cols: cols, eps: eps
            )

            enc.setComputePipelineState(dualPipeline)
            enc.setBuffer(kWeightsBuf, offset: 0, index: 0)
            enc.setBuffer(kScalesBuf, offset: 0, index: 1)
            enc.setBuffer(kBiasesBuf, offset: 0, index: 2)
            enc.setBuffer(vWeightsBuf, offset: 0, index: 3)
            enc.setBuffer(vScalesBuf, offset: 0, index: 4)
            enc.setBuffer(vBiasesBuf, offset: 0, index: 5)
            enc.setBuffer(normOutputBuf, offset: 0, index: 6)
            enc.setBuffer(kOutputBuf, offset: 0, index: 7)
            enc.setBuffer(vOutputBuf, offset: 0, index: 8)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 9)
            dispatchAffineMatvec(encoder: enc, rows: rows)
        }

        // Block layout: K outputs followed by V outputs, mirroring the
        // CPU reference's [K..., V...] concatenation so element-wise
        // comparison aligns without index gymnastics.
        let kPtr = kOutputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        let vPtr = vOutputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        let kOut = Array(UnsafeBufferPointer(start: kPtr, count: rows))
        let vOut = Array(UnsafeBufferPointer(start: vPtr, count: rows))
        return kOut + vOut
    }

    // MARK: - apply_rope (in-place K/Q rotation)
    //
    // After K and V projections (the dual_matvec region above), K and Q
    // get rotated by RoPE before attention. The kernel
    // (`apply_rope`, catalog case 15, kernel at attention.metal:11)
    // mutates the [numHeads, headDim] vector in place using a
    // precomputed cos/sin pair table. The kernel supports three
    // production layouts:
    //   - layout 0: adjacent pairs (Qwen, Llama 2)
    //   - layout 1: split-half (Llama 3, GPT-NeoX style)
    //   - layout 2: proportional split-half (full-attention,
    //     wired up in SmeltLab/SmeltLabTool.swift)

    /// RoPE is two FMAs per output element with no reduction chain —
    /// byte-equal between FP16-rounded CPU and FP16 GPU is the realistic
    /// contract. A loose tolerance would silently absorb real bugs (e.g.
    /// swapped c0/c1 indices on a near-identity rotation table).
    private static let ropeRotationTolerance: Float = 1e-4

    /// Rotation layouts the kernel supports. Values match the integer
    /// constants both kernels read: `attention.metal:37-55` (decode
    /// apply_rope) and `prefill_rope_kv.metal:19-34` (prefill fused
    /// rope+cache).
    private enum RopeLayout: UInt32 {
        case adjacentPairs = 0
        case splitHalf = 1
        case proportionalSplitHalf = 2
    }

    func testApplyRope_MatchesReference_AdjacentPairsLayout() throws {
        try runApplyRopeRegion(numHeads: 4, headDim: 128, ropeDim: 64, layout: .adjacentPairs)
    }

    func testApplyRope_MatchesReference_SplitHalfLayout() throws {
        try runApplyRopeRegion(numHeads: 4, headDim: 128, ropeDim: 64, layout: .splitHalf)
    }

    func testApplyRope_MatchesReference_ProportionalSplitHalfLayout() throws {
        try runApplyRopeRegion(numHeads: 4, headDim: 128, ropeDim: 64, layout: .proportionalSplitHalf)
    }

    private func runApplyRopeRegion(
        numHeads: Int,
        headDim: Int,
        ropeDim: Int,
        layout: RopeLayout
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "attention.metal", functionName: "apply_rope"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile apply_rope pipeline")

        let inputs = makeApplyRopeInputs(
            numHeads: numHeads, headDim: headDim, ropeDim: ropeDim, seed: 1
        )

        let cpuExpected = referenceApplyRopeCPU(
            inputs: inputs,
            numHeads: numHeads, headDim: headDim, ropeDim: ropeDim, layout: layout
        )

        let gpuActual = try runApplyRopeGPU(
            pipeline: pipeline!, inputs: inputs,
            numHeads: numHeads, headDim: headDim, ropeDim: ropeDim, layout: layout
        )

        // Per-head trip-wire: a kernel that rotated only some heads
        // (e.g. an off-by-one head dispatch bound) would still clear a
        // single combined max-abs check because the untouched heads
        // carry the input magnitude through. Assert each head's output
        // is non-trivial.
        var gpuMaxAbs: Float = 0
        for head in 0..<numHeads {
            let start = head * headDim
            let end = start + headDim
            let headMaxAbs = gpuActual[start..<end].map { abs(Float($0)) }.max() ?? 0
            XCTAssertGreaterThan(
                headMaxAbs, Self.tripwireMinAbs,
                "head \(head) output is suspiciously close to zero"
            )
            if headMaxAbs > gpuMaxAbs { gpuMaxAbs = headMaxAbs }
        }

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "apply_rope",
            shape: "[h\(numHeads),d\(headDim),r\(ropeDim),layout=\(layout.rawValue)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(maxDiff, Self.ropeRotationTolerance, "apply_rope diverged from CPU reference (layout=\(layout.rawValue))")
    }

    private struct ApplyRopeInputs {
        let qk: [Float16]
        let cosVal: [Float16]
        let sinVal: [Float16]
    }

    private func makeApplyRopeInputs(
        numHeads: Int,
        headDim: Int,
        ropeDim: Int,
        seed: Int
    ) -> ApplyRopeInputs {
        let qkSeed = Float(seed &* RegionSeed.primary)
        let cosSeed = Float(seed &* RegionSeed.secondary)
        let sinSeed = Float(seed &* RegionSeed.tertiary)

        // qk values cover the full [numHeads, headDim] vector. Centered
        // bias keeps RoPE outputs comfortably above silu's underflow
        // cliff (matters when this gets composed with downstream
        // nonlinear regions).
        let qk = makeFloat16Vector(
            count: numHeads * headDim, seedOffset: qkSeed, multiplier: 0.5, bias: 0.1
        )
        // Rotation tables — the kernel reads ropeDim entries from each.
        // Production tables are derived from theta * position; for the
        // test, any cos/sin pair exercises the rotation math the same
        // way. Keep magnitudes inside [-1, 1] so |c±s| ≤ √2.
        let cosVal = makeFloat16Vector(
            count: ropeDim, seedOffset: cosSeed, multiplier: 0.9, phase: 0.0237
        )
        let sinVal = makeFloat16Vector(
            count: ropeDim, seedOffset: sinSeed, multiplier: 0.9, phase: 0.0193
        )
        return ApplyRopeInputs(qk: qk, cosVal: cosVal, sinVal: sinVal)
    }

    private func referenceApplyRopeCPU(
        inputs: ApplyRopeInputs,
        numHeads: Int,
        headDim: Int,
        ropeDim: Int,
        layout: RopeLayout
    ) -> [Float] {
        let halfRope = ropeDim / 2
        var qk = inputs.qk
        for head in 0..<numHeads {
            let offset = head * headDim
            applyRopeInPlace(
                vec: &qk, offset: offset,
                cos: inputs.cosVal, sin: inputs.sinVal,
                cosBase: 0, sinBase: 0,
                halfRope: halfRope, headDim: headDim, layout: layout
            )
        }
        return qk.map { Float($0) }
    }

    private func runApplyRopeGPU(
        pipeline: MTLComputePipelineState,
        inputs: ApplyRopeInputs,
        numHeads: Int,
        headDim: Int,
        ropeDim: Int,
        layout: RopeLayout
    ) throws -> [Float16] {
        let qkBuf = try makeSharedBuffer(device: device, inputs.qk)
        let cosBuf = try makeSharedBuffer(device: device, inputs.cosVal)
        let sinBuf = try makeSharedBuffer(device: device, inputs.sinVal)

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(qkBuf, offset: 0, index: 0)
            enc.setBuffer(cosBuf, offset: 0, index: 1)
            enc.setBuffer(sinBuf, offset: 0, index: 2)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 3)
            var ropeDimVal = UInt32(ropeDim)
            enc.setBytes(&ropeDimVal, length: 4, index: 4)
            var numHeadsVal = UInt32(numHeads)
            enc.setBytes(&numHeadsVal, length: 4, index: 5)
            var layoutVal = layout.rawValue
            enc.setBytes(&layoutVal, length: 4, index: 6)

            // One thread per (head, dim). The kernel does its own bounds
            // check + halfRope short-circuit, so any threadgroup size
            // ≥ 1 works.
            let total = numHeads * headDim
            let tgWidth = 64
            enc.dispatchThreadgroups(
                MTLSize(width: (total + tgWidth - 1) / tgWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
        }

        let outPtr = qkBuf.contents().bindMemory(to: Float16.self, capacity: numHeads * headDim)
        return Array(UnsafeBufferPointer(start: outPtr, count: numHeads * headDim))
    }

    // MARK: - kv_cache_update (write new K/V at position)
    //
    // After RoPE rotates the new K vector, kv_cache_update copies it
    // into the rolling KV cache at the current position so future
    // attention dispatches can read it. Same kernel writes V (no
    // rotation needed for V). The kernel is a pure memory copy with
    // no arithmetic — byte-equal between input and the touched row of
    // the cache, with every other byte of the cache preserved.

    func testKVCacheUpdate_GPUMatchesReference() throws {
        try runKVCacheUpdateRegion(
            numKVHeads: 4, headDim: 128, cacheSeqCapacity: 256, position: 73
        )
    }

    /// Position 0 exercises the start-of-cache write. seqLen-1 (last
    /// usable position) exercises the end-of-cache write.
    func testKVCacheUpdate_GPUMatchesReference_BoundaryPositions() throws {
        try runKVCacheUpdateRegion(
            numKVHeads: 4, headDim: 128, cacheSeqCapacity: 256, position: 0
        )
        try runKVCacheUpdateRegion(
            numKVHeads: 4, headDim: 128, cacheSeqCapacity: 256, position: 255
        )
    }

    private func runKVCacheUpdateRegion(
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        position: Int
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "attention.metal", functionName: "kv_cache_update"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile kv_cache_update pipeline")

        let inputs = makeKVCacheUpdateInputs(
            numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, seed: 1
        )

        // Trip-wire first: if newKV is near zero then expected == initial
        // cache and a no-op kernel would compare-equal. Verify the
        // fixture has signal before drawing conclusions about kernel
        // correctness.
        let touchedMaxAbs = inputs.newKV.map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(touchedMaxAbs, Self.tripwireMinAbs, "newKV is suspiciously close to zero")

        let cpuExpected = referenceKVCacheUpdateCPU(
            inputs: inputs,
            numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, position: position
        )

        let gpuActual = try runKVCacheUpdateGPU(
            pipeline: pipeline!, inputs: inputs,
            numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, position: position
        )

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        var mismatchCount = 0
        var firstMismatchIdx = -1
        for i in 0..<gpuActual.count where gpuActual[i] != cpuExpected[i] {
            if firstMismatchIdx < 0 { firstMismatchIdx = i }
            mismatchCount += 1
            let diff = abs(Float(gpuActual[i]) - Float(cpuExpected[i]))
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "kv_cache_update",
            shape: "[h\(numKVHeads),d\(headDim),cap\(cacheSeqCapacity),pos\(position)]",
            maxDiff: maxDiff, gpuMaxAbs: touchedMaxAbs
        )
        // Pure memory copy — any drift means either the target row got
        // wrong bytes or an unrelated row was clobbered. Report the
        // first mismatch's expected/actual values so a regression is
        // diagnosable from the test log alone.
        if mismatchCount > 0 {
            XCTFail(
                "\(mismatchCount) cache cells differ; first at index"
                    + " \(firstMismatchIdx): expected \(cpuExpected[firstMismatchIdx]),"
                    + " got \(gpuActual[firstMismatchIdx])"
            )
        }
    }

    private struct KVCacheUpdateInputs {
        let initialCache: [Float16]
        let newKV: [Float16]
    }

    private func makeKVCacheUpdateInputs(
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        seed: Int
    ) -> KVCacheUpdateInputs {
        let cacheCount = numKVHeads * cacheSeqCapacity * headDim
        let initialCache = makeFloat16Vector(
            count: cacheCount, seedOffset: Float(seed &* 11), multiplier: 0.5
        )
        // Different seed offset and a positive bias: the touched row has
        // to observably differ from initial cache contents (otherwise
        // even a no-op kernel would clear the byte-equal check).
        let newKV = makeFloat16Vector(
            count: numKVHeads * headDim,
            seedOffset: Float(seed &* 13), multiplier: 0.5, bias: 0.1
        )
        return KVCacheUpdateInputs(initialCache: initialCache, newKV: newKV)
    }

    private func referenceKVCacheUpdateCPU(
        inputs: KVCacheUpdateInputs,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        position: Int
    ) -> [Float16] {
        // Clone initial cache, overwrite [head, position, *] rows with
        // newKV. Layout matches kernel: [head][seq][dim] stride.
        var expected = inputs.initialCache
        for head in 0..<numKVHeads {
            for dim in 0..<headDim {
                let cacheIdx = head * cacheSeqCapacity * headDim + position * headDim + dim
                expected[cacheIdx] = inputs.newKV[head * headDim + dim]
            }
        }
        return expected
    }

    private func runKVCacheUpdateGPU(
        pipeline: MTLComputePipelineState,
        inputs: KVCacheUpdateInputs,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        position: Int
    ) throws -> [Float16] {
        let cacheCount = numKVHeads * cacheSeqCapacity * headDim
        let cacheBuf = try makeSharedBuffer(device: device, inputs.initialCache)
        let newKVBuf = try makeSharedBuffer(device: device, inputs.newKV)

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(cacheBuf, offset: 0, index: 0)
            enc.setBuffer(newKVBuf, offset: 0, index: 1)
            var cacheCapVal = UInt32(cacheSeqCapacity)
            enc.setBytes(&cacheCapVal, length: 4, index: 2)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 3)
            var posVal = UInt32(position)
            enc.setBytes(&posVal, length: 4, index: 4)
            var numHeadsVal = UInt32(numKVHeads)
            enc.setBytes(&numHeadsVal, length: 4, index: 5)

            // One thread per (head, dim). The kernel does its own bounds
            // check, so any threadgroup size ≥ 1 works.
            let total = numKVHeads * headDim
            let tgWidth = 64
            enc.dispatchThreadgroups(
                MTLSize(width: (total + tgWidth - 1) / tgWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
        }

        let cachePtr = cacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheCount)
        return Array(UnsafeBufferPointer(start: cachePtr, count: cacheCount))
    }

    // MARK: - rope_kv_cache_update (decode K RoPE + cache write)

    func testRopeKVCacheUpdateFusedMatchesStagedGPU() throws {
        try runRopeKVCacheUpdateFusedRegion(
            numKVHeads: 3, headDim: 16, ropeDim: 8,
            cacheSeqCapacity: 7, position: 3, layout: .adjacentPairs
        )
        try runRopeKVCacheUpdateFusedRegion(
            numKVHeads: 3, headDim: 16, ropeDim: 8,
            cacheSeqCapacity: 7, position: 3, layout: .splitHalf
        )
        try runRopeKVCacheUpdateFusedRegion(
            numKVHeads: 3, headDim: 16, ropeDim: 8,
            cacheSeqCapacity: 7, position: 3, layout: .proportionalSplitHalf
        )
    }

    private func runRopeKVCacheUpdateFusedRegion(
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        cacheSeqCapacity: Int,
        position: Int,
        layout: RopeLayout
    ) throws {
        let ropePipeline = makeComputePipeline(
            device: device, shaderFile: "attention.metal", functionName: "apply_rope"
        )
        let cachePipeline = makeComputePipeline(
            device: device, shaderFile: "attention.metal", functionName: "kv_cache_update"
        )
        let fusedPipeline = makeKernelLabAttentionPipeline(functionName: "rope_kv_cache_update")
        try XCTSkipIf(ropePipeline == nil, "Could not compile apply_rope pipeline")
        try XCTSkipIf(cachePipeline == nil, "Could not compile kv_cache_update pipeline")
        try XCTSkipIf(fusedPipeline == nil, "Could not compile rope_kv_cache_update pipeline")

        let ropeInputs = makeApplyRopeInputs(
            numHeads: numKVHeads, headDim: headDim, ropeDim: ropeDim,
            seed: 7 + Int(layout.rawValue)
        )
        let cacheInputs = makeKVCacheUpdateInputs(
            numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, seed: 11 + Int(layout.rawValue)
        )

        let staged = try runStagedRopeKVCacheUpdateGPU(
            ropePipeline: ropePipeline!,
            cachePipeline: cachePipeline!,
            initialCache: cacheInputs.initialCache,
            newKV: ropeInputs.qk,
            cosVal: ropeInputs.cosVal,
            sinVal: ropeInputs.sinVal,
            numKVHeads: numKVHeads,
            headDim: headDim,
            ropeDim: ropeDim,
            cacheSeqCapacity: cacheSeqCapacity,
            position: position,
            layout: layout
        )
        let fused = try runFusedRopeKVCacheUpdateGPU(
            pipeline: fusedPipeline!,
            initialCache: cacheInputs.initialCache,
            newKV: ropeInputs.qk,
            cosVal: ropeInputs.cosVal,
            sinVal: ropeInputs.sinVal,
            numKVHeads: numKVHeads,
            headDim: headDim,
            ropeDim: ropeDim,
            cacheSeqCapacity: cacheSeqCapacity,
            position: position,
            layout: layout
        )

        let gpuMaxAbs = assertPerCacheTripwire(
            cache: fused,
            kvHeads: numKVHeads, cacheSeqCapacity: cacheSeqCapacity,
            headDim: headDim, startPos: position, batchSize: 1,
            label: "fused decode K cache"
        )
        let maxDiff = maxAbsDiff(staged, fused)
        logRegionExactness(
            name: "rope_kv_cache_update",
            shape: "[h\(numKVHeads),d\(headDim),r\(ropeDim),cap\(cacheSeqCapacity),pos\(position),layout=\(layout.rawValue)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )

        if staged != fused {
            let firstMismatch = zip(staged.indices, zip(staged, fused))
                .first { _, pair in pair.0 != pair.1 }
            if let firstMismatch {
                XCTFail(
                    "rope_kv_cache_update differs from staged GPU at index"
                        + " \(firstMismatch.0): expected \(firstMismatch.1.0),"
                        + " got \(firstMismatch.1.1)"
                )
            } else {
                XCTFail("rope_kv_cache_update differs from staged GPU")
            }
        }
    }

    private func makeKernelLabAttentionPipeline(
        functionName: String
    ) -> MTLComputePipelineState? {
        guard let attentionSource = loadMetalShaderSource("attention.metal"),
              let library = try? device.makeLibrary(
                source: attentionSource,
                options: nil
              ),
              let function = library.makeFunction(name: functionName)
        else {
            return nil
        }
        return try? device.makeComputePipelineState(function: function)
    }

    private func runStagedRopeKVCacheUpdateGPU(
        ropePipeline: MTLComputePipelineState,
        cachePipeline: MTLComputePipelineState,
        initialCache: [Float16],
        newKV: [Float16],
        cosVal: [Float16],
        sinVal: [Float16],
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        cacheSeqCapacity: Int,
        position: Int,
        layout: RopeLayout
    ) throws -> [Float16] {
        let cacheBuf = try makeSharedBuffer(device: device, initialCache)
        let newKVBuf = try makeSharedBuffer(device: device, newKV)
        let cosBuf = try makeSharedBuffer(device: device, cosVal)
        let sinBuf = try makeSharedBuffer(device: device, sinVal)

        try runOnGPU(queue: queue) { enc in
            encodeApplyRope(
                encoder: enc,
                pipeline: ropePipeline,
                data: newKVBuf,
                cos: cosBuf,
                sin: sinBuf,
                numHeads: numKVHeads,
                headDim: headDim,
                ropeDim: ropeDim,
                layout: layout
            )
            encodeKVCacheUpdate(
                encoder: enc,
                pipeline: cachePipeline,
                cache: cacheBuf,
                newKV: newKVBuf,
                numHeads: numKVHeads,
                headDim: headDim,
                cacheSeqCapacity: cacheSeqCapacity,
                position: position
            )
        }

        let cacheCount = initialCache.count
        let cachePtr = cacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheCount)
        return Array(UnsafeBufferPointer(start: cachePtr, count: cacheCount))
    }

    private func runFusedRopeKVCacheUpdateGPU(
        pipeline: MTLComputePipelineState,
        initialCache: [Float16],
        newKV: [Float16],
        cosVal: [Float16],
        sinVal: [Float16],
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        cacheSeqCapacity: Int,
        position: Int,
        layout: RopeLayout
    ) throws -> [Float16] {
        let cacheBuf = try makeSharedBuffer(device: device, initialCache)
        let newKVBuf = try makeSharedBuffer(device: device, newKV)
        let cosBuf = try makeSharedBuffer(device: device, cosVal)
        let sinBuf = try makeSharedBuffer(device: device, sinVal)

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(cacheBuf, offset: 0, index: 0)
            enc.setBuffer(newKVBuf, offset: 0, index: 1)
            enc.setBuffer(cosBuf, offset: 0, index: 2)
            enc.setBuffer(sinBuf, offset: 0, index: 3)
            var cacheCapVal = UInt32(cacheSeqCapacity)
            enc.setBytes(&cacheCapVal, length: 4, index: 4)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 5)
            var posVal = UInt32(position)
            enc.setBytes(&posVal, length: 4, index: 6)
            var numHeadsVal = UInt32(numKVHeads)
            enc.setBytes(&numHeadsVal, length: 4, index: 7)
            var ropeDimVal = UInt32(ropeDim)
            enc.setBytes(&ropeDimVal, length: 4, index: 8)
            var layoutVal = layout.rawValue
            enc.setBytes(&layoutVal, length: 4, index: 9)

            let total = numKVHeads * headDim
            enc.dispatchThreads(
                MTLSize(width: total, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(total, 64), height: 1, depth: 1)
            )
        }

        let cacheCount = initialCache.count
        let cachePtr = cacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheCount)
        return Array(UnsafeBufferPointer(start: cachePtr, count: cacheCount))
    }

    private func encodeApplyRope(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        data: MTLBuffer,
        cos: MTLBuffer,
        sin: MTLBuffer,
        numHeads: Int,
        headDim: Int,
        ropeDim: Int,
        layout: RopeLayout
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(data, offset: 0, index: 0)
        encoder.setBuffer(cos, offset: 0, index: 1)
        encoder.setBuffer(sin, offset: 0, index: 2)
        var headDimVal = UInt32(headDim)
        encoder.setBytes(&headDimVal, length: 4, index: 3)
        var ropeDimVal = UInt32(ropeDim)
        encoder.setBytes(&ropeDimVal, length: 4, index: 4)
        var numHeadsVal = UInt32(numHeads)
        encoder.setBytes(&numHeadsVal, length: 4, index: 5)
        var layoutVal = layout.rawValue
        encoder.setBytes(&layoutVal, length: 4, index: 6)

        let total = numHeads * headDim
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 64), height: 1, depth: 1)
        )
    }

    private func encodeKVCacheUpdate(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        cache: MTLBuffer,
        newKV: MTLBuffer,
        numHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        position: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(cache, offset: 0, index: 0)
        encoder.setBuffer(newKV, offset: 0, index: 1)
        var cacheCapVal = UInt32(cacheSeqCapacity)
        encoder.setBytes(&cacheCapVal, length: 4, index: 2)
        var headDimVal = UInt32(headDim)
        encoder.setBytes(&headDimVal, length: 4, index: 3)
        var posVal = UInt32(position)
        encoder.setBytes(&posVal, length: 4, index: 4)
        var numHeadsVal = UInt32(numHeads)
        encoder.setBytes(&numHeadsVal, length: 4, index: 5)

        let total = numHeads * headDim
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 64), height: 1, depth: 1)
        )
    }

    // MARK: - attention_decode (single-token softmax attention with GQA)
    //
    // The decode-time attention kernel
    // (`attention_decode`, catalog case 17, kernel at attention.metal:102):
    // for each query head, compute scaled QK^T over the cache, softmax,
    // weighted V sum. GQA groups gqaRatio = numQHeads / numKVHeads
    // queries per KV head. Sliding-window attention drops cache
    // positions older than (seqLen - slidingWindow).

    /// Softmax + multi-step FP32 reductions push tolerance above pure
    /// matvec — simd_max and simd_sum reorder FMAs vs the CPU's
    /// serial sums. Observed: 0 for full causal, ~4e-6 for sliding
    /// window. 1e-4 leaves headroom for shape variants without
    /// silently absorbing per-head sign-flip bugs.
    private static let attentionDecodeTolerance: Float = 1e-4
    /// attention_prefill drift varies by dispatch shape. Generic 256-tgs
    /// is byte-equal for some shapes (full causal, sliding window) but
    /// drifts ~1e-4 for others (startPos=0). Llama fast path drifts
    /// ~5e-4. Generic-with-tgs=64 (Llama shape + sliding window forcing
    /// fast-path bypass) drifts ~2e-4. 1e-3 covers all observed
    /// reorder noise while still catching per-head GQA-wiring bugs
    /// (those produce O(magnitude) diffs ≥ 1e-2).
    private static let attentionPrefillTolerance: Float = 1e-3
    /// Long-context attention dilutes per-cell magnitudes — softmax over
    /// causalLen positions averages V toward mean(V), so individual
    /// cells can be O(0.005) at causalLen=512 even when global output
    /// max-abs is O(2). The per-cell trip-wire still has to fire on
    /// "kernel skipped this cell" (writes 0) without false-failing on
    /// small-but-valid outputs. 1e-3 is the right line for prefill.
    private static let attentionPrefillCellTripwire: Float = 1e-3

    func testAttentionDecode_FullCausal_MatchesReference() throws {
        try runAttentionDecodeRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            seqLen: 32, slidingWindow: 0, threadgroupWidth: 64
        )
    }

    func testAttentionDecode_SlidingWindow_MatchesReference() throws {
        try runAttentionDecodeRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            seqLen: 32, slidingWindow: 8, threadgroupWidth: 64
        )
    }

    /// Production uses 256-thread threadgroups (8 simdgroups). The
    /// kernel reduces simd_max / simd_sum into tgMax[8] / tgSum[8] —
    /// at tgs=64 only indices 0..1 are touched. This shape exercises
    /// the full 8-way reduction path so a bug in the second-stage
    /// reduction loop bound or simd_group index can't hide behind
    /// the smaller-tgs tests above. seqLen=192 keeps every simdgroup
    /// loaded (s = seqStart + tid steps tgs at a time).
    func testAttentionDecode_LargeThreadgroup_MatchesReference() throws {
        try runAttentionDecodeRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            seqLen: 192, slidingWindow: 0, threadgroupWidth: 256
        )
    }

    private func runAttentionDecodeRegion(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        seqLen: Int,
        slidingWindow: Int,
        threadgroupWidth: Int
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile attention_decode pipeline")

        let cacheSeqCapacity = max(seqLen, 64)
        let inputs = makeAttentionDecodeInputs(
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, cacheSeqCapacity: cacheSeqCapacity, seqLen: seqLen, seed: 1
        )

        let cpuExpected = referenceAttentionDecodeCPU(
            inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, cacheSeqCapacity: cacheSeqCapacity,
            seqLen: seqLen, slidingWindow: slidingWindow
        )

        let gpuActual = try runAttentionDecodeGPU(
            pipeline: pipeline!, inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, cacheSeqCapacity: cacheSeqCapacity,
            seqLen: seqLen, slidingWindow: slidingWindow,
            threadgroupWidth: threadgroupWidth
        )

        // Per-head trip-wire — same defense as apply_rope.
        var gpuMaxAbs: Float = 0
        for head in 0..<numQHeads {
            let start = head * headDim
            let end = start + headDim
            let headMaxAbs = gpuActual[start..<end].map { abs(Float($0)) }.max() ?? 0
            XCTAssertGreaterThan(
                headMaxAbs, Self.tripwireMinAbs,
                "Q head \(head) output is suspiciously close to zero"
            )
            if headMaxAbs > gpuMaxAbs { gpuMaxAbs = headMaxAbs }
        }

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "attention_decode",
            shape: "[qH\(numQHeads),kvH\(numKVHeads),d\(headDim),s\(seqLen),win\(slidingWindow),tgs\(threadgroupWidth)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(maxDiff, Self.attentionDecodeTolerance, "attention_decode diverged from CPU reference")
    }

    private struct AttentionDecodeInputs {
        let query: [Float16]
        let kCache: [Float16]
        let vCache: [Float16]
        let attnMask: [Float16]
    }

    private func makeAttentionDecodeInputs(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        seqLen: Int,
        seed: Int
    ) -> AttentionDecodeInputs {
        let qSeed = Float(seed &* RegionSeed.primary)
        let kSeedBase = Float(seed &* RegionSeed.secondary)
        let vSeedBase = Float(seed &* RegionSeed.tertiary)

        let query = makeFloat16Vector(
            count: numQHeads * headDim, seedOffset: qSeed, multiplier: 0.3, bias: 0.05
        )
        // K and V caches per-kvHead use distinct multipliers so the
        // heads have observably different output magnitudes — a bug
        // that swapped the GQA grouping (kvHead = qHead / gqaRatio)
        // would land Q heads on the wrong KV cluster and surface as
        // a per-head magnitude diff.
        var kCache = [Float16](repeating: 0, count: numKVHeads * cacheSeqCapacity * headDim)
        var vCache = [Float16](repeating: 0, count: numKVHeads * cacheSeqCapacity * headDim)
        for kvHead in 0..<numKVHeads {
            let perHeadCount = cacheSeqCapacity * headDim
            let kHeadVec = makeFloat16Vector(
                count: perHeadCount,
                seedOffset: kSeedBase + Float(kvHead * 1009),
                multiplier: 0.2 + 0.3 * Float(kvHead),
                bias: 0.05
            )
            let vHeadVec = makeFloat16Vector(
                count: perHeadCount,
                seedOffset: vSeedBase + Float(kvHead * 1013),
                multiplier: 0.2 + 0.4 * Float(kvHead),
                bias: 0.05
            )
            for i in 0..<perHeadCount {
                kCache[kvHead * perHeadCount + i] = kHeadVec[i]
                vCache[kvHead * perHeadCount + i] = vHeadVec[i]
            }
        }
        let attnMask = [Float16](repeating: 0, count: 1)
        return AttentionDecodeInputs(
            query: query, kCache: kCache, vCache: vCache, attnMask: attnMask
        )
    }

    private func referenceAttentionDecodeCPU(
        inputs: AttentionDecodeInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        seqLen: Int,
        slidingWindow: Int
    ) -> [Float] {
        let scale = 1 / Float(headDim).squareRoot()
        let gqaRatio = numQHeads / numKVHeads
        let kvStride = cacheSeqCapacity * headDim

        var seqStart = 0
        if slidingWindow > 0 && slidingWindow < seqLen {
            seqStart = seqLen - slidingWindow
        }

        var output = [Float](repeating: 0, count: numQHeads * headDim)

        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            let qOffset = qHead * headDim

            // Snapshot Q in FP32 like the kernel's qShared.
            var q = [Float](repeating: 0, count: headDim)
            for d in 0..<headDim { q[d] = Float(inputs.query[qOffset + d]) }

            // Scaled QK^T scores per cache position, max for numerical
            // stability.
            var scores = [Float](repeating: 0, count: seqLen - seqStart)
            var maxScore: Float = -.infinity
            for s in seqStart..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += q[d] * Float(inputs.kCache[kvHead * kvStride + s * headDim + d])
                }
                let score = dot * scale
                scores[s - seqStart] = score
                if score > maxScore { maxScore = score }
            }

            // exp(score - max), normalize.
            var sumExp: Float = 0
            for i in 0..<scores.count {
                scores[i] = exp(scores[i] - maxScore)
                sumExp += scores[i]
            }
            let invSum = 1 / sumExp

            // Weighted V sum, FP32 accumulation, FP16 round at output.
            for d in 0..<headDim {
                var acc: Float = 0
                for s in seqStart..<seqLen {
                    let w = scores[s - seqStart] * invSum
                    acc += w * Float(inputs.vCache[kvHead * kvStride + s * headDim + d])
                }
                output[qOffset + d] = Float(Float16(acc))
            }
        }
        return output
    }

    private func runAttentionDecodeGPU(
        pipeline: MTLComputePipelineState,
        inputs: AttentionDecodeInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        seqLen: Int,
        slidingWindow: Int,
        threadgroupWidth: Int
    ) throws -> [Float16] {
        let queryBuf = try makeSharedBuffer(device: device, inputs.query)
        let kCacheBuf = try makeSharedBuffer(device: device, inputs.kCache)
        let vCacheBuf = try makeSharedBuffer(device: device, inputs.vCache)
        let attnMaskBuf = try makeSharedBuffer(device: device, inputs.attnMask)
        let outputBuf = try makeSharedBuffer(
            device: device, count: numQHeads * headDim, of: Float16.self
        )

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(queryBuf, offset: 0, index: 0)
            enc.setBuffer(kCacheBuf, offset: 0, index: 1)
            enc.setBuffer(vCacheBuf, offset: 0, index: 2)
            enc.setBuffer(attnMaskBuf, offset: 0, index: 3)
            enc.setBuffer(outputBuf, offset: 0, index: 4)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 5)
            var cacheCapVal = UInt32(cacheSeqCapacity)
            enc.setBytes(&cacheCapVal, length: 4, index: 6)
            var seqLenVal = UInt32(seqLen)
            enc.setBytes(&seqLenVal, length: 4, index: 7)
            var numKVHeadsVal = UInt32(numKVHeads)
            enc.setBytes(&numKVHeadsVal, length: 4, index: 8)
            var scaleVal = 1 / Float(headDim).squareRoot()
            enc.setBytes(&scaleVal, length: 4, index: 9)
            var slidingWindowVal = UInt32(slidingWindow)
            enc.setBytes(&slidingWindowVal, length: 4, index: 10)

            // Dispatch contract: numQHeads threadgroups × tgs threads.
            // Kernel reduces simd_max / simd_sum across simdgroups, with
            // tgMax/tgSum sized 8 → tgs must be ≤ 256 (8 simdgroups).
            enc.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            )
        }

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self, capacity: numQHeads * headDim
        )
        return Array(UnsafeBufferPointer(start: outPtr, count: numQHeads * headDim))
    }

    // MARK: - attention_prefill (block causal attention with GQA)
    //
    // Multi-token version of attention_decode: for each position in the
    // query block, attend over the cached prefix [0, startPos + pos]
    // (with sliding-window override). Output layout has chunk position
    // as the outermost dimension to match batched-matmul downstream
    // consumers. Kernel at `prefill_attention.metal:13`. The kernel has
    // a 64-thread fast-path specialization for the headDim=64 hot
    // path (32 Q heads, 8 KV heads); other shapes — and
    // Llama-shaped dispatches that fail the `seqStart == 0` predicate —
    // fall through to the generic reduction tree (8 simdgroups at
    // tgs=256, 2 at tgs=64).

    func testAttentionPrefill_FullCausal_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 4, slidingWindow: 0,
            threadgroupWidth: 256
        )
    }

    func testAttentionPrefill_SlidingWindow_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 4, slidingWindow: 8,
            threadgroupWidth: 256
        )
    }

    /// Single-position chunk. Exercises the `tgid.y == 0` only dispatch
    /// path — a kernel that wrote nothing for any tgid.y > 0 cell would
    /// be invisible at B=1, but a kernel that confused queryPos and
    /// failed to emit anything for tgid.y=0 would zero the output, which
    /// the per-cell trip-wire catches.
    func testAttentionPrefill_BatchSizeOne_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 1, startPos: 4, slidingWindow: 0,
            threadgroupWidth: 256
        )
    }

    /// startPos=0 (no prefix). `causalLen = queryPos + 1`. A bug that
    /// drops the `+1` in `causalLen = startPos + queryPos + 1` would
    /// land queryPos=0 with causalLen=0, producing zeros. The per-cell
    /// trip-wire catches that; the equality assertion catches more
    /// subtle off-by-ones that still produce nonzero output.
    func testAttentionPrefill_StartPosZero_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 0, slidingWindow: 0,
            threadgroupWidth: 256
        )
    }

    /// headDim=64 hot path: 32 Q heads, 8 KV heads,
    /// 64-thread threadgroup, causalLen ≤ 512. Hits the fast-path
    /// specialization at `prefill_attention.metal:54` — a different
    /// reduction structure (one threadgroup loops over headDim threads
    /// to cover all score positions, with `partial[2]` for cross-simd
    /// reduction).
    func testAttentionPrefill_LlamaHotPath_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 32, numKVHeads: 8, headDim: 64,
            batchSize: 8, startPos: 10, slidingWindow: 0,
            threadgroupWidth: 64
        )
    }

    /// Llama hot-path shape with sliding window forcing `seqStart != 0`.
    /// The fast-path predicate at `prefill_attention.metal:54` includes
    /// `seqStart == 0`, so the kernel must fall through to the generic
    /// path here. A regression that drops the `seqStart == 0` predicate
    /// would silently use the fast path and skip prefix tokens.
    func testAttentionPrefill_LlamaSlidingWindow_FallsThroughToGeneric() throws {
        try runAttentionPrefillRegion(
            numQHeads: 32, numKVHeads: 8, headDim: 64,
            batchSize: 8, startPos: 10, slidingWindow: 4,
            threadgroupWidth: 64
        )
    }

    /// Fast-path's `causalLen <= 512` boundary. The kernel sizes its
    /// `scores[maxSeq=512]` threadgroup buffer for that limit; a
    /// regression in the score-cache bound (e.g. `scores[s]` indexed
    /// past 511) would corrupt threadgroup memory.
    func testAttentionPrefill_LlamaCausalLenBoundary_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 32, numKVHeads: 8, headDim: 64,
            batchSize: 8, startPos: 504, slidingWindow: 0,
            threadgroupWidth: 64
        )
    }

    // MARK: - attention_prefill_softcap (logit softcap variant)
    //
    // Adds the logit softcap `score = c * tanh(score / c)` after
    // the scaled dot product, before softmax. Kernel at
    // `prefill_attention.metal:414`. Routes through the same
    // attention_prefill runner with an optional softcap parameter —
    // same dispatch and reduction structure, just one extra constant
    // at buffer 11. Production softcap is ~50; tight values exercise
    // the tanh nonlinearity directly.

    func testAttentionPrefillSoftcap_FullCausal_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 4, slidingWindow: 0,
            threadgroupWidth: 256, softcap: 50.0
        )
    }

    func testAttentionPrefillSoftcap_SlidingWindow_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 4, slidingWindow: 8,
            threadgroupWidth: 256, softcap: 50.0
        )
    }

    /// softcap=0.05 forces the tanh transform out of its identity
    /// region. Raw scores at fixture magnitudes are O(0.03–0.07);
    /// at softcap=50 (production) or even softcap=2 the tanh acts
    /// as near-identity and a kernel that omitted softcap would
    /// still pass. softcap=0.05 puts score/c in roughly [-0.6, 1.4]
    /// where tanh observably compresses scores by 10–30%.
    /// Empirically the post-softcap output magnitudes drop from
    /// ~0.33 (softcap=50) to ~0.14 (softcap=0.05), confirming the
    /// clamp is actively transforming.
    func testAttentionPrefillSoftcap_TightSoftcap_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 4, slidingWindow: 0,
            threadgroupWidth: 256, softcap: 0.05
        )
    }

    /// activeLen > 512 forces the kernel out of the fast path
    /// (predicate requires `activeLen <= 512` at the top of the
    /// kernel) into the generic three-pass branch. The generic path
    /// repeats the `softcap * tanh(score/c)` invocation across
    /// score-cache compute, score-cache reload, and output
    /// accumulation phases — without this test the duplicated
    /// softcap math is uncovered.
    func testAttentionPrefillSoftcap_GenericPath_MatchesReference() throws {
        try runAttentionPrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128,
            batchSize: 8, startPos: 600, slidingWindow: 0,
            threadgroupWidth: 256, softcap: 50.0
        )
    }

    private func runAttentionPrefillRegion(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        batchSize: Int,
        startPos: Int,
        slidingWindow: Int,
        threadgroupWidth: Int,
        softcap: Float? = nil
    ) throws {
        let functionName = softcap == nil
            ? "attention_prefill" : "attention_prefill_softcap"
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "prefill_attention.metal",
            functionName: functionName
        )
        try XCTSkipIf(pipeline == nil, "Could not compile \(functionName) pipeline")

        let cacheSeqCapacity = max(startPos + batchSize, 64)
        let inputs = makeAttentionPrefillInputs(
            numQHeads: numQHeads, numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, batchSize: batchSize, seed: 1
        )

        let cpuExpected = referenceAttentionPrefillCPU(
            inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, batchSize: batchSize,
            startPos: startPos, slidingWindow: slidingWindow,
            softcap: softcap
        )

        let gpuActual = try runAttentionPrefillGPU(
            pipeline: pipeline!, inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads, headDim: headDim,
            cacheSeqCapacity: cacheSeqCapacity, batchSize: batchSize,
            startPos: startPos, slidingWindow: slidingWindow,
            threadgroupWidth: threadgroupWidth,
            softcap: softcap
        )

        // Per-(position, qHead) trip-wire: a kernel that wrote correctly
        // to one batch position but skipped another (e.g. a tgid.y bound
        // bug) would otherwise clear a single combined max-abs check.
        // Threshold is `attentionPrefillCellTripwire`, not the shared
        // `tripwireMinAbs` — see that constant's docstring.
        var gpuMaxAbs: Float = 0
        for pos in 0..<batchSize {
            for qHead in 0..<numQHeads {
                let start = (pos * numQHeads + qHead) * headDim
                let end = start + headDim
                let cellMaxAbs = gpuActual[start..<end].map { abs(Float($0)) }.max() ?? 0
                XCTAssertGreaterThan(
                    cellMaxAbs, Self.attentionPrefillCellTripwire,
                    "[pos \(pos), qHead \(qHead)] output is suspiciously close to zero"
                )
                if cellMaxAbs > gpuMaxAbs { gpuMaxAbs = cellMaxAbs }
            }
        }

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        let softcapTag = softcap.map { ",softcap\($0)" } ?? ""
        logRegionExactness(
            name: functionName,
            shape: "[qH\(numQHeads),kvH\(numKVHeads),d\(headDim),B\(batchSize),start\(startPos),win\(slidingWindow),tgs\(threadgroupWidth)\(softcapTag)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(
            maxDiff, Self.attentionPrefillTolerance,
            "\(functionName) diverged from CPU reference"
        )
    }

    private struct AttentionPrefillInputs {
        let queries: [Float16]   // [B, numQHeads, headDim]
        let kCache: [Float16]    // [numKVHeads, cacheSeqCapacity, headDim]
        let vCache: [Float16]    // [numKVHeads, cacheSeqCapacity, headDim]
    }

    private func makeAttentionPrefillInputs(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        batchSize: Int,
        seed: Int
    ) -> AttentionPrefillInputs {
        let qSeed = Float(seed &* RegionSeed.primary)
        let kSeedBase = Float(seed &* RegionSeed.secondary)
        let vSeedBase = Float(seed &* RegionSeed.tertiary)

        let queries = makeFloat16Vector(
            count: batchSize * numQHeads * headDim,
            seedOffset: qSeed, multiplier: 0.3, bias: 0.05
        )
        // Per-kvHead distinct multipliers — same defense as
        // attention_decode. A swapped GQA grouping would land Q heads
        // on the wrong KV cluster and surface as a per-head magnitude
        // diff, not silently within tolerance.
        var kCache = [Float16](repeating: 0, count: numKVHeads * cacheSeqCapacity * headDim)
        var vCache = [Float16](repeating: 0, count: numKVHeads * cacheSeqCapacity * headDim)
        for kvHead in 0..<numKVHeads {
            let perHeadCount = cacheSeqCapacity * headDim
            let kHeadVec = makeFloat16Vector(
                count: perHeadCount,
                seedOffset: kSeedBase + Float(kvHead * 1009),
                multiplier: 0.2 + 0.3 * Float(kvHead), bias: 0.05
            )
            let vHeadVec = makeFloat16Vector(
                count: perHeadCount,
                seedOffset: vSeedBase + Float(kvHead * 1013),
                multiplier: 0.2 + 0.4 * Float(kvHead), bias: 0.05
            )
            for i in 0..<perHeadCount {
                kCache[kvHead * perHeadCount + i] = kHeadVec[i]
                vCache[kvHead * perHeadCount + i] = vHeadVec[i]
            }
        }
        return AttentionPrefillInputs(queries: queries, kCache: kCache, vCache: vCache)
    }

    private func referenceAttentionPrefillCPU(
        inputs: AttentionPrefillInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        batchSize: Int,
        startPos: Int,
        slidingWindow: Int,
        softcap: Float? = nil
    ) -> [Float] {
        let scale = 1 / Float(headDim).squareRoot()
        let gqaRatio = numQHeads / numKVHeads
        let kvStride = cacheSeqCapacity * headDim
        let qStride = numQHeads * headDim
        // Pre-convert FP16 → Float once. The inner Q-K dot product
        // and V accumulation each read the cache headDim times per
        // (pos, qHead, s); at GenericPath shape (B=8, qHeads=4,
        // activeLen~600, headDim=128) the hoist drops ~5M widens
        // out of the hot path.
        let kCacheF = inputs.kCache.map { Float($0) }
        let vCacheF = inputs.vCache.map { Float($0) }

        var output = [Float](repeating: 0, count: batchSize * numQHeads * headDim)

        for pos in 0..<batchSize {
            // Causal mask varies per chunk position; sliding window
            // overrides only when the active window is shorter than
            // the causal prefix.
            let causalLen = startPos + pos + 1
            var seqStart = 0
            if slidingWindow > 0 && slidingWindow < causalLen {
                seqStart = causalLen - slidingWindow
            }
            let activeLen = causalLen - seqStart

            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                let qOffset = pos * qStride + qHead * headDim

                var q = [Float](repeating: 0, count: headDim)
                for d in 0..<headDim { q[d] = Float(inputs.queries[qOffset + d]) }

                var scores = [Float](repeating: 0, count: activeLen)
                var maxScore: Float = -.infinity
                for s in seqStart..<causalLen {
                    var dot: Float = 0
                    let kBase = kvHead * kvStride + s * headDim
                    for d in 0..<headDim {
                        dot += q[d] * kCacheF[kBase + d]
                    }
                    var score = dot * scale
                    // Logit softcap: score = c * tanh(score / c).
                    if let c = softcap {
                        score = c * tanh(score / c)
                    }
                    scores[s - seqStart] = score
                    if score > maxScore { maxScore = score }
                }

                var sumExp: Float = 0
                for i in 0..<scores.count {
                    scores[i] = exp(scores[i] - maxScore)
                    sumExp += scores[i]
                }
                let invSum = 1 / sumExp

                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in seqStart..<causalLen {
                        let w = scores[s - seqStart] * invSum
                        acc += w * vCacheF[kvHead * kvStride + s * headDim + d]
                    }
                    output[qOffset + d] = Float(Float16(acc))
                }
            }
        }
        return output
    }

    private func runAttentionPrefillGPU(
        pipeline: MTLComputePipelineState,
        inputs: AttentionPrefillInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        cacheSeqCapacity: Int,
        batchSize: Int,
        startPos: Int,
        slidingWindow: Int,
        threadgroupWidth: Int,
        softcap: Float? = nil
    ) throws -> [Float16] {
        let queriesBuf = try makeSharedBuffer(device: device, inputs.queries)
        let kCacheBuf = try makeSharedBuffer(device: device, inputs.kCache)
        let vCacheBuf = try makeSharedBuffer(device: device, inputs.vCache)
        let outputBuf = try makeSharedBuffer(
            device: device, count: batchSize * numQHeads * headDim, of: Float16.self
        )

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(queriesBuf, offset: 0, index: 0)
            enc.setBuffer(kCacheBuf, offset: 0, index: 1)
            enc.setBuffer(vCacheBuf, offset: 0, index: 2)
            enc.setBuffer(outputBuf, offset: 0, index: 3)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 4)
            var seqLenVal = UInt32(batchSize)
            enc.setBytes(&seqLenVal, length: 4, index: 5)
            var startPosVal = UInt32(startPos)
            enc.setBytes(&startPosVal, length: 4, index: 6)
            var cacheSeqVal = UInt32(cacheSeqCapacity)
            enc.setBytes(&cacheSeqVal, length: 4, index: 7)
            var numKVHeadsVal = UInt32(numKVHeads)
            enc.setBytes(&numKVHeadsVal, length: 4, index: 8)
            var scaleVal = 1 / Float(headDim).squareRoot()
            enc.setBytes(&scaleVal, length: 4, index: 9)
            var slidingWindowVal = UInt32(slidingWindow)
            enc.setBytes(&slidingWindowVal, length: 4, index: 10)
            if var softcapVal = softcap {
                enc.setBytes(&softcapVal, length: 4, index: 11)
            }

            // Dispatch contract: (numQHeads, batchSize) threadgroups ×
            // tgs threads. tgs=64 hits the Llama fast path; tgs=256
            // takes the generic 8-simdgroup reduction tree.
            enc.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            )
        }

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * numQHeads * headDim
        )
        return Array(UnsafeBufferPointer(start: outPtr, count: batchSize * numQHeads * headDim))
    }

    // MARK: - fused_lut_matmul (multi-token LUT-quantized matmul)
    //
    // Prefill matmul kernel. Different quantization scheme from
    // affine_matvec: each group of `groupSize` rows shares a 16-entry
    // FP16 LUT, and each weight is a u4 index into that LUT (2 indices
    // packed per byte). Used for prefill Q/K/V/O projections and FFN
    // matmuls. Kernel at `prefill_matmul.metal:13`. Dispatch contract:
    // (R, B) threadgroups × 256 threads, one TG = one output element.
    //
    // The decode-side LUT-matvec equivalent (`fused_lut_matvec`) is
    // catalog case 4; this prefill version is what the package
    // verifier and benchmark path use for batched matmuls.

    /// Pure quantized matmul — FP32 acc + simd_sum reduction across 8
    /// simdgroups. No softmax, no nonlinearity. Empirically byte-equal
    /// (max diff = 0) across all current shapes including cols=2048
    /// and groupSize=64. 5e-3 matches `affineMatvecTolerance` and
    /// would still catch any real bug since real bugs produce
    /// O(magnitude) diffs, not sub-ULP drift.
    private static let lutMatmulTolerance: Float = 5e-3

    func testLutMatmul_AlignedShape_MatchesReference() throws {
        try runLutMatmulRegion(rows: 32, cols: 128, groupSize: 16, batchSize: 4)
    }

    /// batchSize=1 exercises the tgid.y == 0 only dispatch grid. A bug
    /// indexing batch from `tgid.y - 1` or similar would be invisible
    /// at B=1 in the wrong direction but would zero the output, which
    /// the per-(batch, row) trip-wire catches.
    func testLutMatmul_BatchOne_MatchesReference() throws {
        try runLutMatmulRegion(rows: 32, cols: 128, groupSize: 16, batchSize: 1)
    }

    /// Sanity case for moderate cols. At cols=256, halfCols=128, only
    /// threads tid 0..127 enter the dot-product loop (tids 128..255
    /// skip — `j = tid` already exceeds halfCols). The
    /// ProductionHiddenDim shape below covers true strided-loop
    /// stress (4 iterations per thread).
    func testLutMatmul_WiderCols_MatchesReference() throws {
        try runLutMatmulRegion(rows: 64, cols: 256, groupSize: 16, batchSize: 8)
    }

    /// Production hidden dim (2048; Qwen 0.6B class). cols=2048
    /// → halfCols=1024 → 4 iterations per thread, accumulating ~4096
    /// products per row. Tests whether the simd_sum reorder budget
    /// holds at production scale; the smaller shapes are too narrow
    /// to surface reorder error.
    func testLutMatmul_ProductionHiddenDim_MatchesReference() throws {
        try runLutMatmulRegion(rows: 32, cols: 2048, groupSize: 16, batchSize: 4)
    }

    /// groupSize=64 is the affine-path canonical (LUT typically uses
    /// 16) but valid for the kernel and worth coverage — exercises a
    /// less-common `row / groupSize` divisor that downstream models
    /// could elect.
    func testLutMatmul_LargeGroupSize_MatchesReference() throws {
        try runLutMatmulRegion(rows: 128, cols: 128, groupSize: 64, batchSize: 4)
    }

    private func runLutMatmulRegion(
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "prefill_matmul.metal",
            functionName: "fused_lut_matmul"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile fused_lut_matmul pipeline")

        let inputs = makeLutMatmulInputs(
            rows: rows, cols: cols, groupSize: groupSize,
            batchSize: batchSize, seed: 1
        )

        let cpuExpected = referenceLutMatmulCPU(
            inputs: inputs,
            rows: rows, cols: cols, groupSize: groupSize, batchSize: batchSize
        )

        let gpuActual = try runLutMatmulGPU(
            pipeline: pipeline!, inputs: inputs,
            rows: rows, cols: cols, groupSize: groupSize, batchSize: batchSize
        )

        // Per-batch AND per-row trip-wires. Each output element is one
        // scalar (batch, row), so the row-wise check across batches
        // catches a tgid.x off-by-one (writes 0 to row r for all
        // batches → row max = 0) that the per-batch check misses
        // because other rows in the same batch keep batch max nonzero.
        // Single sweep updates both per-axis max arrays.
        var batchMax = [Float](repeating: 0, count: batchSize)
        var rowMax = [Float](repeating: 0, count: rows)
        var gpuMaxAbs: Float = 0
        for b in 0..<batchSize {
            for r in 0..<rows {
                let v = abs(Float(gpuActual[b * rows + r]))
                if v > batchMax[b] { batchMax[b] = v }
                if v > rowMax[r] { rowMax[r] = v }
                if v > gpuMaxAbs { gpuMaxAbs = v }
            }
        }
        for b in 0..<batchSize {
            XCTAssertGreaterThan(
                batchMax[b], Self.tripwireMinAbs,
                "batch \(b) output is suspiciously close to zero"
            )
        }
        for r in 0..<rows {
            XCTAssertGreaterThan(
                rowMax[r], Self.tripwireMinAbs,
                "row \(r) output is suspiciously close to zero across all batches"
            )
        }

        XCTAssertEqual(gpuActual.count, cpuExpected.count)
        var maxDiff: Float = 0
        for i in 0..<gpuActual.count {
            let diff = abs(Float(gpuActual[i]) - cpuExpected[i])
            if diff > maxDiff { maxDiff = diff }
        }
        logRegionExactness(
            name: "fused_lut_matmul",
            shape: "[r\(rows),c\(cols),g\(groupSize),B\(batchSize)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(
            maxDiff, Self.lutMatmulTolerance,
            "fused_lut_matmul diverged from CPU reference"
        )
    }

    private struct LutMatmulInputs {
        let indices: [UInt8]   // [R, C/2] packed u4
        let lut: [Float16]     // [nGroups, 16]
        let input: [Float16]   // [B, C]
    }

    private func makeLutMatmulInputs(
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int,
        seed: Int
    ) -> LutMatmulInputs {
        precondition(rows % groupSize == 0, "rows must be divisible by groupSize")
        let numGroups = rows / groupSize
        let halfCols = cols / 2

        // Packed u4 indices — same scrambling pattern as
        // makeQuantizedAffineFixture's weight bytes.
        let indicesSeed = seed &* 131
        var indices = [UInt8](repeating: 0, count: rows * halfCols)
        for i in 0..<indices.count {
            indices[i] = UInt8(truncatingIfNeeded: i &* 17 &+ 23 &+ indicesSeed)
        }

        // 16 FP16 values per group. Range [-0.4, 0.4] keeps dot-product
        // outputs in FP16 range even at cols=2048 (output magnitudes
        // grow as sqrt(N) for random sums, so ~√2048 × 0.4 × 0.5 ≈
        // 9, well within FP16 max=65504).
        let lutSeed = Float(seed &* RegionSeed.secondary)
        var lut = [Float16](repeating: 0, count: numGroups * 16)
        for i in 0..<lut.count {
            lut[i] = Float16(sin((Float(i) + lutSeed) * 0.0431) * 0.4)
        }

        let inputSeed = Float(seed &* RegionSeed.primary)
        let input = makeFloat16Vector(
            count: batchSize * cols, seedOffset: inputSeed, multiplier: 0.5
        )

        return LutMatmulInputs(indices: indices, lut: lut, input: input)
    }

    private func referenceLutMatmulCPU(
        inputs: LutMatmulInputs,
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) -> [Float] {
        let halfCols = cols / 2
        // Pre-convert FP16 → Float once. At cols=2048, B=4, R=32 the
        // inner loop hits each lut/input cell hundreds of times; the
        // hoist drops ~700K Float() conversions out of the hot path.
        let lutF = inputs.lut.map { Float($0) }
        let inputF = inputs.input.map { Float($0) }
        var output = [Float](repeating: 0, count: batchSize * rows)

        for b in 0..<batchSize {
            for r in 0..<rows {
                let group = r / groupSize
                let lutBase = group * 16
                let rowIdxBase = r * halfCols
                let batchInputBase = b * cols
                var acc: Float = 0
                for j in 0..<halfCols {
                    let packed = inputs.indices[rowIdxBase + j]
                    let lowIdx = Int(packed & 0xF)
                    let highIdx = Int(packed >> 4)
                    let col = j * 2
                    acc += lutF[lutBase + lowIdx] * inputF[batchInputBase + col]
                    acc += lutF[lutBase + highIdx] * inputF[batchInputBase + col + 1]
                }
                output[b * rows + r] = Float(Float16(acc))
            }
        }
        return output
    }

    private func runLutMatmulGPU(
        pipeline: MTLComputePipelineState,
        inputs: LutMatmulInputs,
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) throws -> [Float16] {
        let indicesBuf = try makeSharedBuffer(device: device, inputs.indices)
        let lutBuf = try makeSharedBuffer(device: device, inputs.lut)
        let inputBuf = try makeSharedBuffer(device: device, inputs.input)
        let outputBuf = try makeSharedBuffer(
            device: device, count: batchSize * rows, of: Float16.self
        )

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(indicesBuf, offset: 0, index: 0)
            enc.setBuffer(lutBuf, offset: 0, index: 1)
            enc.setBuffer(inputBuf, offset: 0, index: 2)
            enc.setBuffer(outputBuf, offset: 0, index: 3)
            var colsVal = UInt32(cols)
            enc.setBytes(&colsVal, length: 4, index: 4)
            var gsVal = UInt32(groupSize)
            enc.setBytes(&gsVal, length: 4, index: 5)
            var nrVal = UInt32(rows)
            enc.setBytes(&nrVal, length: 4, index: 6)
            // Dispatch: (R, B) threadgroups × 256 threads, one TG per
            // output element — no tail handling needed since the grid
            // sizes exactly to (rows, batchSize).
            enc.dispatchThreadgroups(
                MTLSize(width: rows, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        return Array(UnsafeBufferPointer(start: outPtr, count: batchSize * rows))
    }

    // MARK: - rope_and_kv_cache_prefill (fused RoPE + cache write)
    //
    // Prefill counterpart to apply_rope + kv_cache_update. Rotates Q
    // and K in place across the entire B-position chunk, then writes
    // rotated K and unrotated V to the KV cache at `absPos = startPos
    // + pos`. Replaces 4×B unrolled dispatches with one. Kernel at
    // `prefill_rope_kv.metal:64`. Dispatch: (B, max(qHeads, kvHeads))
    // threadgroups × headDim threads — heads beyond qHeads/kvHeads
    // are skipped via predicates.

    func testRopeAndKvCachePrefill_AdjacentPairsLayout_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .adjacentPairs
        )
    }

    func testRopeAndKvCachePrefill_SplitHalfLayout_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .splitHalf
        )
    }

    /// Proportional split-half layout. Differs from plain split-half in that the rotated dim
    /// pairs span (pair, pair + headDim/2) rather than (pair, pair +
    /// halfRope), and the cos/sin indices shift to (pair, pair +
    /// halfRope). Production-critical for full-attention models.
    func testRopeAndKvCachePrefill_ProportionalSplitHalfLayout_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .proportionalSplitHalf
        )
    }

    /// Single-position chunk. Exercises the tgid.x=0-only dispatch
    /// edge. A bug that skipped writes for tgid.x=0 (e.g. an `if pos
    /// > 0` guard) would leave the buffer at its input fixture, which
    /// the equality assertion catches; the per-(batch, head) trip-wire
    /// adds belt-and-suspenders for any zeroing variant.
    func testRopeAndKvCachePrefill_BatchOne_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 1, startPos: 8, layout: .adjacentPairs
        )
    }

    /// startPos=0 means absPos = pos. Catches a `cosBase = absPos *
    /// ropeDim - 1` style off-by-one (the cos/sin lookup would land
    /// in the wrong table row for pos=0 only).
    func testRopeAndKvCachePrefill_StartPosZero_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 0, layout: .adjacentPairs
        )
    }

    /// Llama 3 hot path: headDim=64, 32 Q heads, 8 KV heads, GQA 4:1.
    /// 64-thread threadgroup matches headDim, putting reduction at
    /// exactly 2 simdgroups instead of 4. A simdgroup-count-sensitive
    /// bug would only surface at this shape.
    func testRopeAndKvCachePrefill_LlamaHotPath_MatchesReference() throws {
        try runRopeAndKvCachePrefillRegion(
            numQHeads: 32, numKVHeads: 8, headDim: 64, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .splitHalf
        )
    }

    private func runRopeAndKvCachePrefillRegion(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        startPos: Int,
        layout: RopeLayout
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "prefill_rope_kv.metal",
            functionName: "rope_and_kv_cache_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rope_and_kv_cache_prefill pipeline")

        let cacheSeqCapacity = max(startPos + batchSize, 64)
        let maxSeqLen = startPos + batchSize
        let inputs = makeRopeAndKvCachePrefillInputs(
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, maxSeqLen: maxSeqLen,
            cacheSeqCapacity: cacheSeqCapacity, seed: 1
        )

        let cpuExpected = referenceRopeAndKvCachePrefillCPU(
            inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, startPos: startPos,
            cacheSeqCapacity: cacheSeqCapacity, layout: layout
        )

        let gpuActual = try runRopeAndKvCachePrefillGPU(
            pipeline: pipeline!, inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, startPos: startPos,
            cacheSeqCapacity: cacheSeqCapacity, layout: layout
        )

        // Per-output trip-wires. The kernel writes 4 distinct buffers
        // (rotated Q, rotated K, K cache, V cache); a wiring bug that
        // skipped one would clear a single combined max-abs check.
        // Each output has its own slice and its own per-(pos, head)
        // sub-check to catch dispatch-grid axis bugs.
        let qLen = batchSize * numQHeads * headDim
        let kLen = batchSize * numKVHeads * headDim
        let cacheLen = numKVHeads * cacheSeqCapacity * headDim

        var gpuMaxAbs: Float = 0
        gpuMaxAbs = max(gpuMaxAbs, assertPerHeadTripwire(
            slice: gpuActual.q,
            heads: numQHeads, headDim: headDim, batches: batchSize,
            label: "rotated Q"
        ))
        gpuMaxAbs = max(gpuMaxAbs, assertPerHeadTripwire(
            slice: gpuActual.k,
            heads: numKVHeads, headDim: headDim, batches: batchSize,
            label: "rotated K"
        ))
        // Caches use [head][seq][dim] layout — only the touched rows
        // [absPos = startPos + pos] should differ from the initial
        // fixture. The V cache reads the (untouched-by-rotation) V
        // input, so its trip-wire is on input magnitudes.
        gpuMaxAbs = max(gpuMaxAbs, assertPerCacheTripwire(
            cache: gpuActual.kCache,
            kvHeads: numKVHeads, cacheSeqCapacity: cacheSeqCapacity,
            headDim: headDim, startPos: startPos, batchSize: batchSize,
            label: "K cache"
        ))
        gpuMaxAbs = max(gpuMaxAbs, assertPerCacheTripwire(
            cache: gpuActual.vCache,
            kvHeads: numKVHeads, cacheSeqCapacity: cacheSeqCapacity,
            headDim: headDim, startPos: startPos, batchSize: batchSize,
            label: "V cache"
        ))

        XCTAssertEqual(gpuActual.q.count, qLen)
        XCTAssertEqual(gpuActual.k.count, kLen)
        XCTAssertEqual(gpuActual.kCache.count, cacheLen)
        XCTAssertEqual(gpuActual.vCache.count, cacheLen)

        let perOutputDiff: [(name: String, diff: Float)] = [
            ("Q", maxAbsDiff(gpuActual.q, cpuExpected.q)),
            ("K", maxAbsDiff(gpuActual.k, cpuExpected.k)),
            ("kCache", maxAbsDiff(gpuActual.kCache, cpuExpected.kCache)),
            ("vCache", maxAbsDiff(gpuActual.vCache, cpuExpected.vCache)),
        ]
        let worst = perOutputDiff.max { $0.diff < $1.diff }!
        let maxDiff = worst.diff

        logRegionExactness(
            name: "rope_and_kv_cache_prefill",
            shape: "[qH\(numQHeads),kvH\(numKVHeads),d\(headDim),r\(ropeDim),B\(batchSize),start\(startPos),layout=\(layout.rawValue)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        // Same FP16-ULP-tight budget as apply_rope — pure rotation +
        // memcpy, no reduction reorder, byte-equal CPU/GPU expected.
        // Identify the diverged buffer in the failure message —
        // multi-output kernels make the bare maxDiff number useless
        // for triage.
        XCTAssertLessThan(
            maxDiff, Self.ropeRotationTolerance,
            "rope_and_kv_cache_prefill diverged from CPU reference"
                + " (layout=\(layout.rawValue), worst output = \(worst.name)"
                + " at maxDiff = \(maxDiff))"
        )
    }

    private struct RopeAndKvCachePrefillInputs {
        let queries: [Float16]      // [B, qHeads, headDim]
        let keys: [Float16]         // [B, kvHeads, headDim]
        let values: [Float16]       // [B, kvHeads, headDim]
        let cosTable: [Float16]     // [maxSeqLen, ropeDim]
        let sinTable: [Float16]     // [maxSeqLen, ropeDim]
        let initialKCache: [Float16]  // [kvHeads, cacheSeqCapacity, headDim]
        let initialVCache: [Float16]  // [kvHeads, cacheSeqCapacity, headDim]
    }

    private struct RopeAndKvCachePrefillOutputs {
        let q: [Float16]
        let k: [Float16]
        let kCache: [Float16]
        let vCache: [Float16]
    }

    private func makeRopeAndKvCachePrefillInputs(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        maxSeqLen: Int,
        cacheSeqCapacity: Int,
        seed: Int
    ) -> RopeAndKvCachePrefillInputs {
        let qSeed = Float(seed &* RegionSeed.primary)
        let kSeed = Float(seed &* RegionSeed.secondary)
        let vSeed = Float(seed &* RegionSeed.tertiary)
        let cosSeed = Float(seed &* 1009)
        let sinSeed = Float(seed &* 1013)
        let kCacheSeed = Float(seed &* 11)
        let vCacheSeed = Float(seed &* 13)

        let queries = makeFloat16Vector(
            count: batchSize * numQHeads * headDim,
            seedOffset: qSeed, multiplier: 0.5, bias: 0.1
        )
        let keys = makeFloat16Vector(
            count: batchSize * numKVHeads * headDim,
            seedOffset: kSeed, multiplier: 0.5, bias: 0.1
        )
        let values = makeFloat16Vector(
            count: batchSize * numKVHeads * headDim,
            seedOffset: vSeed, multiplier: 0.5, bias: 0.1
        )
        // Rotation tables — the kernel reads ropeDim entries from each.
        // Magnitudes inside [-1, 1] so |c±s| ≤ √2.
        let cosTable = makeFloat16Vector(
            count: maxSeqLen * ropeDim, seedOffset: cosSeed, multiplier: 0.9, phase: 0.0237
        )
        let sinTable = makeFloat16Vector(
            count: maxSeqLen * ropeDim, seedOffset: sinSeed, multiplier: 0.9, phase: 0.0193
        )
        let initialKCache = makeFloat16Vector(
            count: numKVHeads * cacheSeqCapacity * headDim,
            seedOffset: kCacheSeed, multiplier: 0.5
        )
        let initialVCache = makeFloat16Vector(
            count: numKVHeads * cacheSeqCapacity * headDim,
            seedOffset: vCacheSeed, multiplier: 0.5
        )
        return RopeAndKvCachePrefillInputs(
            queries: queries, keys: keys, values: values,
            cosTable: cosTable, sinTable: sinTable,
            initialKCache: initialKCache, initialVCache: initialVCache
        )
    }

    private func referenceRopeAndKvCachePrefillCPU(
        inputs: RopeAndKvCachePrefillInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        startPos: Int,
        cacheSeqCapacity: Int,
        layout: RopeLayout
    ) -> RopeAndKvCachePrefillOutputs {
        var q = inputs.queries
        var k = inputs.keys
        var kCache = inputs.initialKCache
        var vCache = inputs.initialVCache
        let halfRope = ropeDim / 2

        for pos in 0..<batchSize {
            let absPos = startPos + pos
            // Same row index for cos and sin — the kernel reads both
            // tables at `absPos * ropeDim`.
            let tableBase = absPos * ropeDim

            for head in 0..<numQHeads {
                let qOff = pos * numQHeads * headDim + head * headDim
                applyRopeInPlace(
                    vec: &q, offset: qOff,
                    cos: inputs.cosTable, sin: inputs.sinTable,
                    cosBase: tableBase, sinBase: tableBase,
                    halfRope: halfRope, headDim: headDim, layout: layout
                )
            }

            for head in 0..<numKVHeads {
                let kOff = pos * numKVHeads * headDim + head * headDim
                applyRopeInPlace(
                    vec: &k, offset: kOff,
                    cos: inputs.cosTable, sin: inputs.sinTable,
                    cosBase: tableBase, sinBase: tableBase,
                    halfRope: halfRope, headDim: headDim, layout: layout
                )
                let cacheOff = head * cacheSeqCapacity * headDim + absPos * headDim
                let vOff = pos * numKVHeads * headDim + head * headDim
                for d in 0..<headDim {
                    kCache[cacheOff + d] = k[kOff + d]
                    vCache[cacheOff + d] = inputs.values[vOff + d]
                }
            }
        }
        return RopeAndKvCachePrefillOutputs(q: q, k: k, kCache: kCache, vCache: vCache)
    }

    private func applyRopeInPlace(
        vec: inout [Float16],
        offset: Int,
        cos: [Float16], sin: [Float16],
        cosBase: Int, sinBase: Int,
        halfRope: Int, headDim: Int, layout: RopeLayout
    ) {
        for pair in 0..<halfRope {
            let d0: Int
            let d1: Int
            let c0Index: Int
            let c1Index: Int
            switch layout {
            case .adjacentPairs:
                d0 = pair * 2
                d1 = d0 + 1
                c0Index = d0
                c1Index = d1
            case .splitHalf:
                d0 = pair
                d1 = pair + halfRope
                c0Index = d0
                c1Index = d1
            case .proportionalSplitHalf:
                d0 = pair
                d1 = pair + headDim / 2
                c0Index = pair
                c1Index = pair + halfRope
            }
            let xEven = Float(vec[offset + d0])
            let xOdd = Float(vec[offset + d1])
            let cosE = Float(cos[cosBase + c0Index])
            let sinE = Float(sin[sinBase + c0Index])
            let cosO = Float(cos[cosBase + c1Index])
            let sinO = Float(sin[sinBase + c1Index])
            vec[offset + d0] = Float16(xEven * cosE - xOdd * sinE)
            vec[offset + d1] = Float16(xOdd * cosO + xEven * sinO)
        }
    }

    /// Shared GPU dispatch for `rope_and_kv_cache_prefill` (rmsEps =
    /// nil) and `fused_norm_rope_and_kv_cache_prefill` (rmsEps set).
    /// Buffer layout 0..14 is identical between the two kernels;
    /// rmsEps at index 15 is only set when the fused-norm variant
    /// pipeline is dispatched.
    private func runRopeAndKvCachePrefillGPU(
        pipeline: MTLComputePipelineState,
        inputs: RopeAndKvCachePrefillInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        startPos: Int,
        cacheSeqCapacity: Int,
        layout: RopeLayout,
        rmsEps: Float? = nil
    ) throws -> RopeAndKvCachePrefillOutputs {
        let queriesBuf = try makeSharedBuffer(device: device, inputs.queries)
        let keysBuf = try makeSharedBuffer(device: device, inputs.keys)
        let valuesBuf = try makeSharedBuffer(device: device, inputs.values)
        let cosBuf = try makeSharedBuffer(device: device, inputs.cosTable)
        let sinBuf = try makeSharedBuffer(device: device, inputs.sinTable)
        let kCacheBuf = try makeSharedBuffer(device: device, inputs.initialKCache)
        let vCacheBuf = try makeSharedBuffer(device: device, inputs.initialVCache)

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(queriesBuf, offset: 0, index: 0)
            enc.setBuffer(keysBuf, offset: 0, index: 1)
            enc.setBuffer(valuesBuf, offset: 0, index: 2)
            enc.setBuffer(cosBuf, offset: 0, index: 3)
            enc.setBuffer(sinBuf, offset: 0, index: 4)
            enc.setBuffer(kCacheBuf, offset: 0, index: 5)
            enc.setBuffer(vCacheBuf, offset: 0, index: 6)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 7)
            var ropeDimVal = UInt32(ropeDim)
            enc.setBytes(&ropeDimVal, length: 4, index: 8)
            var qHeadsVal = UInt32(numQHeads)
            enc.setBytes(&qHeadsVal, length: 4, index: 9)
            var kvHeadsVal = UInt32(numKVHeads)
            enc.setBytes(&kvHeadsVal, length: 4, index: 10)
            var seqLenVal = UInt32(batchSize)
            enc.setBytes(&seqLenVal, length: 4, index: 11)
            var startPosVal = UInt32(startPos)
            enc.setBytes(&startPosVal, length: 4, index: 12)
            var cacheCapVal = UInt32(cacheSeqCapacity)
            enc.setBytes(&cacheCapVal, length: 4, index: 13)
            var layoutVal = layout.rawValue
            enc.setBytes(&layoutVal, length: 4, index: 14)
            if var rmsEpsVal = rmsEps {
                enc.setBytes(&rmsEpsVal, length: 4, index: 15)
            }

            // Dispatch contract: (B, max(qHeads, kvHeads)) threadgroups
            // × headDim threads. Heads beyond qHeads/kvHeads are
            // skipped via predicates inside the kernel.
            let maxHeads = max(numQHeads, numKVHeads)
            enc.dispatchThreadgroups(
                MTLSize(width: batchSize, height: maxHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
        }

        let qLen = batchSize * numQHeads * headDim
        let kLen = batchSize * numKVHeads * headDim
        let cacheLen = numKVHeads * cacheSeqCapacity * headDim
        let qPtr = queriesBuf.contents().bindMemory(to: Float16.self, capacity: qLen)
        let kPtr = keysBuf.contents().bindMemory(to: Float16.self, capacity: kLen)
        let kCachePtr = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheLen)
        let vCachePtr = vCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheLen)
        return RopeAndKvCachePrefillOutputs(
            q: Array(UnsafeBufferPointer(start: qPtr, count: qLen)),
            k: Array(UnsafeBufferPointer(start: kPtr, count: kLen)),
            kCache: Array(UnsafeBufferPointer(start: kCachePtr, count: cacheLen)),
            vCache: Array(UnsafeBufferPointer(start: vCachePtr, count: cacheLen))
        )
    }

    /// Max absolute diff between two FP16 arrays, computed in FP32.
    private func maxAbsDiff(_ a: [Float16], _ b: [Float16]) -> Float {
        precondition(a.count == b.count)
        var maxDiff: Float = 0
        for i in 0..<a.count {
            let d = abs(Float(a[i]) - Float(b[i]))
            if d > maxDiff { maxDiff = d }
        }
        return maxDiff
    }

    // MARK: - fused_norm_rope_and_kv_cache_prefill (V-norm + RoPE + cache)
    //
    // Further-fused version of rope_and_kv_cache_prefill: adds V RMS
    // scaling (`rsqrt(mean(V²) + eps) * V`, no learned gain weight)
    // on the way to the cache. Reuses RopeAndKvCachePrefillInputs/
    // Outputs and the merged GPU dispatch helper since the buffer
    // layout is identical. Kernel at `prefill_rope_kv.metal:189`,
    // standard path. The kvHeads==0 Q-only norm path (kernel:226) is
    // uncovered — exotic mode, deferred until production usage lands.

    /// V RMS scaling adds a simd_sum reduction over headDim threads,
    /// reordering FMAs vs the CPU's serial sum. Observed: 0 at
    /// generic 128-thread shapes, ~1e-3 on the Llama hot path
    /// (headDim=64 → 2 simdgroups → tighter loop where reduction
    /// reorder matters more relative to V magnitudes). 1e-3 absorbs
    /// the observed drift while still catching real bugs (those
    /// surface at O(magnitude) on at least one of the four output
    /// buffers).
    private static let fusedNormRopeAndKvTolerance: Float = 1e-3

    func testFusedNormRopeAndKvCachePrefill_AdjacentPairsLayout_MatchesReference() throws {
        try runFusedNormRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .adjacentPairs
        )
    }

    func testFusedNormRopeAndKvCachePrefill_SplitHalfLayout_MatchesReference() throws {
        try runFusedNormRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .splitHalf
        )
    }

    /// Proportional split-half layout — the production-critical case for this kernel
    /// since full-attention models prefer the proportional
    /// split.
    func testFusedNormRopeAndKvCachePrefill_ProportionalSplitHalfLayout_MatchesReference() throws {
        try runFusedNormRopeAndKvCachePrefillRegion(
            numQHeads: 4, numKVHeads: 2, headDim: 128, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .proportionalSplitHalf
        )
    }

    /// Llama 3 hot path: headDim=64 → 64-thread threadgroup → 2
    /// simdgroups instead of 4. The V-norm reduction iterates `for s
    /// = 0..<nSimds` (kernel:381), so 2 vs 4 simdgroups exercises a
    /// different reduction-loop bound — the most likely place for a
    /// regression in this newly-added V-norm path.
    func testFusedNormRopeAndKvCachePrefill_LlamaHotPath_MatchesReference() throws {
        try runFusedNormRopeAndKvCachePrefillRegion(
            numQHeads: 32, numKVHeads: 8, headDim: 64, ropeDim: 64,
            batchSize: 4, startPos: 8, layout: .splitHalf
        )
    }

    private func runFusedNormRopeAndKvCachePrefillRegion(
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        startPos: Int,
        layout: RopeLayout
    ) throws {
        let pipeline = makeComputePipeline(
            device: device, shaderFile: "prefill_rope_kv.metal",
            functionName: "fused_norm_rope_and_kv_cache_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile fused_norm_rope_and_kv_cache_prefill pipeline")

        let cacheSeqCapacity = max(startPos + batchSize, 64)
        let maxSeqLen = startPos + batchSize
        let rmsEps: Float = 1e-6
        let inputs = makeRopeAndKvCachePrefillInputs(
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, maxSeqLen: maxSeqLen,
            cacheSeqCapacity: cacheSeqCapacity, seed: 1
        )

        let cpuExpected = referenceFusedNormRopeAndKvCachePrefillCPU(
            inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, startPos: startPos,
            cacheSeqCapacity: cacheSeqCapacity, layout: layout, rmsEps: rmsEps
        )

        let gpuActual = try runRopeAndKvCachePrefillGPU(
            pipeline: pipeline!, inputs: inputs,
            numQHeads: numQHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeDim: ropeDim,
            batchSize: batchSize, startPos: startPos,
            cacheSeqCapacity: cacheSeqCapacity, layout: layout, rmsEps: rmsEps
        )

        var gpuMaxAbs: Float = 0
        gpuMaxAbs = max(gpuMaxAbs, assertPerHeadTripwire(
            slice: gpuActual.q,
            heads: numQHeads, headDim: headDim, batches: batchSize,
            label: "rotated Q"
        ))
        gpuMaxAbs = max(gpuMaxAbs, assertPerHeadTripwire(
            slice: gpuActual.k,
            heads: numKVHeads, headDim: headDim, batches: batchSize,
            label: "rotated K"
        ))
        gpuMaxAbs = max(gpuMaxAbs, assertPerCacheTripwire(
            cache: gpuActual.kCache,
            kvHeads: numKVHeads, cacheSeqCapacity: cacheSeqCapacity,
            headDim: headDim, startPos: startPos, batchSize: batchSize,
            label: "K cache"
        ))
        gpuMaxAbs = max(gpuMaxAbs, assertPerCacheTripwire(
            cache: gpuActual.vCache,
            kvHeads: numKVHeads, cacheSeqCapacity: cacheSeqCapacity,
            headDim: headDim, startPos: startPos, batchSize: batchSize,
            label: "V cache"
        ))

        let perOutputDiff: [(name: String, diff: Float)] = [
            ("Q", maxAbsDiff(gpuActual.q, cpuExpected.q)),
            ("K", maxAbsDiff(gpuActual.k, cpuExpected.k)),
            ("kCache", maxAbsDiff(gpuActual.kCache, cpuExpected.kCache)),
            ("vCache", maxAbsDiff(gpuActual.vCache, cpuExpected.vCache)),
        ]
        let worst = perOutputDiff.max { $0.diff < $1.diff }!
        let maxDiff = worst.diff

        logRegionExactness(
            name: "fused_norm_rope_and_kv_cache_prefill",
            shape: "[qH\(numQHeads),kvH\(numKVHeads),d\(headDim),r\(ropeDim),B\(batchSize),start\(startPos),layout=\(layout.rawValue)]",
            maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs
        )
        XCTAssertLessThan(
            maxDiff, Self.fusedNormRopeAndKvTolerance,
            "fused_norm_rope_and_kv_cache_prefill diverged from CPU reference"
                + " (layout=\(layout.rawValue), worst output = \(worst.name)"
                + " at maxDiff = \(maxDiff))"
        )
    }

    private func referenceFusedNormRopeAndKvCachePrefillCPU(
        inputs: RopeAndKvCachePrefillInputs,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        ropeDim: Int,
        batchSize: Int,
        startPos: Int,
        cacheSeqCapacity: Int,
        layout: RopeLayout,
        rmsEps: Float
    ) -> RopeAndKvCachePrefillOutputs {
        var q = inputs.queries
        var k = inputs.keys
        var kCache = inputs.initialKCache
        var vCache = inputs.initialVCache
        let halfRope = ropeDim / 2
        // Pre-convert V to Float once. Each (pos, head) reads V twice
        // per dim — sumSq + scale write — so this halves FP16→Float
        // conversions in the hot path; same precedent as lut_matmul.
        let valuesF = inputs.values.map { Float($0) }

        for pos in 0..<batchSize {
            let absPos = startPos + pos
            let tableBase = absPos * ropeDim

            for head in 0..<numQHeads {
                let qOff = pos * numQHeads * headDim + head * headDim
                applyRopeInPlace(
                    vec: &q, offset: qOff,
                    cos: inputs.cosTable, sin: inputs.sinTable,
                    cosBase: tableBase, sinBase: tableBase,
                    halfRope: halfRope, headDim: headDim, layout: layout
                )
            }

            for head in 0..<numKVHeads {
                let kOff = pos * numKVHeads * headDim + head * headDim
                applyRopeInPlace(
                    vec: &k, offset: kOff,
                    cos: inputs.cosTable, sin: inputs.sinTable,
                    cosBase: tableBase, sinBase: tableBase,
                    halfRope: halfRope, headDim: headDim, layout: layout
                )
                let cacheOff = head * cacheSeqCapacity * headDim + absPos * headDim
                let vOff = pos * numKVHeads * headDim + head * headDim
                for d in 0..<headDim {
                    kCache[cacheOff + d] = k[kOff + d]
                }
                // V RMS scaling: rs = rsqrt(mean(V²) + eps); cache =
                // V * rs (no learned weight in this kernel variant).
                // Sequential here vs the kernel's interleaving with
                // K-cache write is provably equivalent: V is read-only
                // (`device const half*` at kernel:192), so reordering
                // the V-norm loop relative to the K-cache write
                // produces the same result.
                var sumSq: Float = 0
                for d in 0..<headDim {
                    let v = valuesF[vOff + d]
                    sumSq += v * v
                }
                let mean = sumSq / Float(headDim)
                let rs = 1 / (mean + rmsEps).squareRoot()
                for d in 0..<headDim {
                    vCache[cacheOff + d] = Float16(valuesF[vOff + d] * rs)
                }
            }
        }
        return RopeAndKvCachePrefillOutputs(q: q, k: k, kCache: kCache, vCache: vCache)
    }

}
