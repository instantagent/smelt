// CompilerTestSupport — Shared scaffolding for SmeltCompilerTests
// targets that build real .smeltpkg artifacts from synthetic local
// weights. Each test suite owns a managed temp root so concurrent
// test runs don't collide and crash-cleanup runs at process exit.

import Darwin
import Foundation
@testable import SmeltCompiler
import SmeltModels
import SmeltSchema

/// The authored registry IR for a fixture, keyed by module id (a legacy `.cam`
/// or a `.module.json` suffix is tolerated as a label). The `.cam` grammar was
/// deleted in Phase C; `ModuleAuthoringParityTests` now pins each definition
/// byte-for-byte against the checked-in `Models/<id>.module.json`, so this is a
/// drop-in for tests that previously loaded a grammar fixture.
func registryModuleIR(_ name: String) -> SmeltCAMIR {
    let id = name.hasSuffix(".module.json") ? String(name.dropLast(".module.json".count))
        : name.hasSuffix(".cam") ? String(name.dropLast(".cam".count)) : name
    guard let ir = SmeltModels.definition(id: id) else {
        preconditionFailure("no authored registry module for '\(name)' (id '\(id)')")
    }
    return ir
}

/// Apply a JSON-value mutation to an authored module IR and re-decode it — the
/// grammar-free equivalent of loading a fixture, editing its `.cam` text, and
/// reparsing. The module JSON *is* the lowered IR, so editing the lowered field
/// reproduces the same drift the text edit encoded, then `decodeModule`'s
/// `validated()` re-runs on the result.
func mutatedModuleIR(
    _ base: SmeltCAMIR,
    _ mutate: (inout [String: Any]) throws -> Void
) throws -> SmeltCAMIR {
    let data = try base.canonicalJSONData()
    guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        preconditionFailure("module IR did not encode to a JSON object")
    }
    try mutate(&object)
    let mutatedData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return try JSONDecoder().decode(SmeltCAMIR.self, from: mutatedData).validated()
}

/// Walk up from the test source's compile-time path until a file
/// matching `relPath` exists. Returns the absolute path. Throws if
/// the file isn't found within 8 ancestor steps.
///
/// Lets tests reach repo-rooted artifacts (specs in `tools/`,
/// scripts under `tools/`) without depending on the test runner's
/// current working directory, which differs between `swift test`
/// and Xcode invocations.
func repoRelativePath(_ relPath: String) throws -> String {
    if FileManager.default.fileExists(atPath: relPath) {
        return URL(fileURLWithPath: relPath).path
    }
    var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0 ..< 8 {
        let candidate = dir.appendingPathComponent(relPath).path
        if FileManager.default.fileExists(atPath: candidate) {
            return candidate
        }
        let parent = dir.deletingLastPathComponent()
        if parent == dir { break }
        dir = parent
    }
    throw NSError(
        domain: "CompilerTestSupport.repoRelativePath", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "could not locate \(relPath)"]
    )
}

/// Create a process-unique temp root under `$TMPDIR` and register an
/// `atexit` block to remove it. Suite-scoped: each suite calls this
/// once at file load and reuses the URL across its tests.
func makeManagedTempRoot(_ name: String) -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(name)-\(getpid())", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try? FileManager.default.createDirectory(
        at: root, withIntermediateDirectories: true
    )
    atexit_b {
        try? FileManager.default.removeItem(at: root)
    }
    return root
}

/// Produce a unique subdirectory under `root`, one per test invocation.
/// Lets concurrent runs share the suite root without colliding on
/// weights.bin paths.
func makeTempDir(under root: URL) throws -> URL {
    let dir = root.appendingPathComponent(
        UUID().uuidString, isDirectory: true
    )
    try FileManager.default.createDirectory(
        at: dir, withIntermediateDirectories: true
    )
    return dir
}

/// Emit `weights.json` describing `layout` into `dir`. Internal
/// helper shared by `writeLocalWeightsBundle` (zero-fill) and
/// `writeRandomWeightsBundle` (random-fill); both versions share
/// the manifest schema and only differ on the `weights.bin`
/// payload.
private func writeWeightsManifestJSON(
    layout: [SmeltWeightEntry], into dir: URL
) throws {
    var manifest: [String: [String: Any]] = [:]
    for entry in layout {
        var meta: [String: Any] = [
            "offset": Int(entry.offset),
            "sizeBytes": Int(entry.sizeBytes),
            "shape": entry.shape,
            "quantized": entry.dtype == .u4Lut
                || entry.dtype == .affineU4
                || entry.dtype == .turboQuantH,
            "dtype": entry.dtype.rawValue,
        ]
        if let groupSize = entry.groupSize { meta["groupSize"] = groupSize }
        if let lutOffset = entry.lutOffset {
            meta["lutOffset"] = Int(lutOffset)
        }
        if let lutSizeBytes = entry.lutSizeBytes {
            meta["lutSizeBytes"] = Int(lutSizeBytes)
        }
        if let packedRowStride = entry.packedRowStride {
            meta["packedRowStride"] = packedRowStride
        }
        if let paddedCols = entry.paddedCols {
            meta["paddedCols"] = paddedCols
        }
        if let scalesOffset = entry.scalesOffset {
            meta["scalesOffset"] = Int(scalesOffset)
        }
        if let scalesSizeBytes = entry.scalesSizeBytes {
            meta["scalesSizeBytes"] = Int(scalesSizeBytes)
        }
        if let biasesOffset = entry.biasesOffset {
            meta["biasesOffset"] = Int(biasesOffset)
        }
        if let biasesSizeBytes = entry.biasesSizeBytes {
            meta["biasesSizeBytes"] = Int(biasesSizeBytes)
        }
        if let codebookOffset = entry.codebookOffset {
            meta["codebookOffset"] = Int(codebookOffset)
        }
        if let codebookSizeBytes = entry.codebookSizeBytes {
            meta["codebookSizeBytes"] = Int(codebookSizeBytes)
        }
        manifest[entry.name] = meta
    }
    let weightsJSON = try JSONSerialization.data(
        withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]
    )
    try weightsJSON.write(to: dir.appendingPathComponent("weights.json"))
}

/// Write a synthetic local-weights bundle with seeded-random fp16
/// values, valid identity permutations for any int32 index buffers,
/// and zero-filled u4/affine slots. Produces `weights.json` +
/// `weights.bin` compatible with `SmeltCompiler.build(weightsDir:)`.
///
/// Use over `writeLocalWeightsBundle` when a test needs to drive
/// `decodeStep` numerically — zero-filled weights make RMS norm
/// produce NaN, which propagates through every dispatch downstream.
/// Random fp16 values keep activations finite.
///
/// Random distribution: uniform in `[-0.05, 0.05)` (roughly fan-in
/// scale for a 1024-dim hidden), small enough to keep matvec outputs
/// finite under fp16 even after summing across hidden dims. Seed
/// `0x5_E11_75EED` reads as "SMELLI SEED" and is non-zero (xorshift
/// collapses on a zero seed).
func writeRandomWeightsBundle(
    layout: [SmeltWeightEntry], into root: URL, seed: UInt64 = 0x5_E11_75EED
) throws -> URL {
    for entry in layout where entry.dtype == .u4Lut || entry.dtype == .affineU4 {
        throw NSError(
            domain: "writeRandomWeightsBundle", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "tensor '\(entry.name)' is u4-quantized; this helper "
                + "cannot synthesise valid packed nibbles + LUT/scales/biases. "
                + "Use a fp16-strategy spec or test against a real package."]
        )
    }

    let dir = try makeTempDir(under: root)
    try writeWeightsManifestJSON(layout: layout, into: dir)

    let totalBytes = Int(SmeltWeightManifestLoader.totalBytes(from: layout))
    var bin = Data(count: totalBytes)

    var rng = seed
    func nextWord() -> UInt64 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return rng
    }

    bin.withUnsafeMutableBytes { rawBuffer in
        guard let base = rawBuffer.baseAddress else { return }
        for entry in layout {
            // sizeBytes==0 marks a placeholder entry the layout
            // emitted but doesn't write data for; respect it
            // independently of the shape product (which can be 1
            // for a rank-0 tensor).
            guard entry.sizeBytes > 0 else { continue }
            let elementCount = entry.shape.reduce(1, *)
            guard elementCount > 0 else { continue }
            let offset = Int(entry.offset)

            switch entry.dtype {
            case .fp16:
                let dst = base.advanced(by: offset)
                    .bindMemory(to: Float16.self, capacity: elementCount)
                for i in 0 ..< elementCount {
                    let bits = nextWord()
                    let unitFloat = Float(bits & 0xFFFF) / Float(0x10000)  // [0,1)
                    dst[i] = Float16(unitFloat * 0.1 - 0.05)               // [-0.05, 0.05)
                }
            case .bf16:
                // bf16 = the high 16 bits of the Float's IEEE-754 pattern.
                let dst = base.advanced(by: offset)
                    .bindMemory(to: UInt16.self, capacity: elementCount)
                for i in 0 ..< elementCount {
                    let bits = nextWord()
                    let unitFloat = Float(bits & 0xFFFF) / Float(0x10000)
                    dst[i] = UInt16((unitFloat * 0.1 - 0.05).bitPattern >> 16)
                }
            case .int32:
                // Index buffers (e.g., masked_embedding_token_ordering).
                // Identity permutation keeps the cluster sparse lm_head
                // from collapsing every cluster onto vocab index 0.
                let dst = base.advanced(by: offset)
                    .bindMemory(to: Int32.self, capacity: elementCount)
                for i in 0 ..< elementCount {
                    dst[i] = Int32(i)
                }
            case .fp32:
                // fp32-trunk norm weights (W0.3). Weight-direct RMS norm
                // (rms_norm_codec_f32) means a ZERO weight would zero the
                // whole trunk on both sides — making the parity memcmp a
                // vacuous 0==0. Fill near 1.0 so non-trivial values flow.
                let dst = base.advanced(by: offset)
                    .bindMemory(to: Float.self, capacity: elementCount)
                for i in 0 ..< elementCount {
                    let bits = nextWord()
                    let unitFloat = Float(bits & 0xFFFF) / Float(0x10000)  // [0,1)
                    dst[i] = 1.0 + (unitFloat * 0.1 - 0.05)                // [0.95, 1.05)
                }
            case .raw, .u4Lut, .affineU4, .turboQuantH, .binary1, .ternary2:
                // Caller-side validation above rejects u4Lut /
                // affineU4 before reaching this loop. Packed quantized layouts
                // require format-aware fixtures rather than dense random fill.
                // .raw layouts have no current consumer in tests that need
                // synthetic numerics.
                continue
            }
        }
    }

    try bin.write(to: dir.appendingPathComponent("weights.bin"))
    return dir
}

/// Write a synthetic local-weights bundle (`weights.json` + zero-filled
/// `weights.bin`) compatible with `SmeltCompiler.build(weightsDir:)`.
/// Used by integration tests that exercise the codegen + manifest path
/// without requiring real model weights.
func writeLocalWeightsBundle(
    layout: [SmeltWeightEntry], into root: URL
) throws -> URL {
    let dir = try makeTempDir(under: root)
    try writeWeightsManifestJSON(layout: layout, into: dir)

    let totalBytes = Int(SmeltWeightManifestLoader.totalBytes(from: layout))
    try Data(count: totalBytes).write(
        to: dir.appendingPathComponent("weights.bin")
    )
    return dir
}

/// Synthetic-weights filling strategy for `buildSyntheticPackageFromSpec`.
/// Random is needed when `decodeStep` is exercised numerically (zeros
/// trip RMS-norm into NaN); local is fine when only the manifest +
/// codegen path is being tested.
enum SyntheticWeights {
    case local
    case random
}

/// Build a synthetic `.smeltpkg` from a programmatic `SmeltModelIR`: write a
/// synthetic-weights bundle in `mode`, then build. Each call gets its own temp
/// output dir under `root` so concurrent tests don't collide on `weights.bin`
/// paths.
func buildSyntheticPackageFromSpec(
    ir: SmeltModelIR,
    inputName: String = "fixture",
    weights: SyntheticWeights = .local,
    into root: URL
) throws -> SmeltCompiler.BuildResult {
    try validateSmeltIR(ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightsDir: URL
    switch weights {
    case .local:
        weightsDir = try writeLocalWeightsBundle(layout: layout, into: root)
    case .random:
        weightsDir = try writeRandomWeightsBundle(layout: layout, into: root)
    }
    let outputDir = try makeTempDir(under: root)
    let result = try managedBuild(
        ir: ir,
        inputName: inputName,
        outputDir: outputDir.path,
        weightsDir: weightsDir.path,
        shaderDir: "Resources/Shaders"
    )
    // The input synthetic-weights bundle is consumed by build() and not needed
    // after — the .smeltpkg is self-contained. Remove it so its weights.bin
    // (the largest temp file) doesn't accumulate across the suite and exhaust
    // the CI runner's disk (only the suite root is atexit-cleaned otherwise).
    try? FileManager.default.removeItem(at: weightsDir)
    return result
}

/// Build a hand-written TTS `.smeltpkg` from a COMPILED fp32-trunk package's own
/// `weights.bin` — the talker-trunk parity-gate pattern. Remaps every weight
/// entry to the `talker.*` checkpoint names the hand `Qwen3TTSGPU` driver expects
/// (bf16 projections raw, fp32 norms), so the compiled and hand paths consume
/// byte-identical weights and the only variable is the compute path. Returns the
/// hand package directory plus the COMPILED manifest (callers read its slot
/// layout to pin the compiled package's rope tables). Shared by the f32-trunk
/// decode (synthetic) and the real-weights decode/prefill parity gates; only
/// `modelName` varies.
func buildHandTalkerPackage(
    compiledPackagePath: String, into root: URL, modelName: String
) throws -> (handPackageDir: String, compiledManifest: SmeltManifest) {
    let manifest = try SmeltManifest.decode(from: Data(contentsOf: URL(
        fileURLWithPath: "\(compiledPackagePath)/manifest.json")))
    let weightsBlob = try Data(contentsOf: URL(
        fileURLWithPath: "\(compiledPackagePath)/weights.bin"))
    let entriesByName = Dictionary(
        uniqueKeysWithValues: manifest.weights.entries.map { ($0.name, $0) })

    func bytes(_ name: String) -> Data {
        let e = entriesByName[name]!
        return weightsBlob.subdata(in: Int(e.offset)..<Int(e.offset + e.sizeBytes))
    }
    func ttsName(_ canonical: String) -> String {
        // layers_0_self_attn_q_proj_weight → talker.model.layers.0.self_attn.q_proj.weight
        if canonical == "norm_weight" { return "talker.model.norm.weight" }
        var n = canonical
        n = n.replacingOccurrences(of: "_self_attn_", with: ".self_attn.")
        n = n.replacingOccurrences(of: "_mlp_", with: ".mlp.")
        n = n.replacingOccurrences(of: "_input_layernorm_weight", with: ".input_layernorm.weight")
        n = n.replacingOccurrences(of: "_post_attention_layernorm_weight", with: ".post_attention_layernorm.weight")
        n = n.replacingOccurrences(of: "layers_", with: "talker.model.layers.")
        n = n.replacingOccurrences(of: "proj_weight", with: "proj.weight")
        n = n.replacingOccurrences(of: "norm_weight", with: "norm.weight")
        return n
    }
    var ttsSpecs: [Qwen3TTSPackageBuilder.WeightSpec] = []
    var ttsPayload: [String: Data] = [:]
    for entry in manifest.weights.entries where entry.name != "embed_tokens" {
        let dtype: Qwen3TTSPackageBuilder.WeightDType = entry.dtype == .bf16 ? .bf16 : .f32
        let name = ttsName(entry.name)
        ttsSpecs.append(.init(name: name, shape: entry.shape, dtype: dtype))
        ttsPayload[name] = bytes(entry.name)
    }
    let ttsDir = try makeTempDir(under: root).appendingPathComponent("hand.smeltpkg").path
    try Qwen3TTSPackageBuilder.build(
        specs: ttsSpecs,
        fill: { spec, _, slice in
            let data = ttsPayload[spec.name]!
            precondition(data.count == spec.dataBytes,
                         "hand-package fill size mismatch for \(spec.name)")
            data.withUnsafeBytes { src in
                slice.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: data.count)
            }
        },
        pipelines: Qwen3TTSPackageBuilder.ttsPipelineNames,
        modelName: modelName,
        eosTokens: [2150],
        shaderDir: "Resources/Shaders",
        outputPath: ttsDir)
    return (ttsDir, manifest)
}

func qwen3TTSTestTensorBlocks(
    for specs: [Qwen3TTSPackageBuilder.WeightSpec]
) -> [String: String] {
    Dictionary(uniqueKeysWithValues: specs.map { spec in
        (spec.name, qwen3TTSTestTensorBlock(for: spec.name))
    })
}

func qwen3TTSTestTensorSourceDTypes(
    for specs: [Qwen3TTSPackageBuilder.WeightSpec]
) -> [String: SmeltPackageSpec.TensorDType] {
    Dictionary(uniqueKeysWithValues: specs.map { spec in
        let block = qwen3TTSTestTensorBlock(for: spec.name)
        return (spec.name, block == "codec-decoder" ? .f32 : .bf16)
    })
}

func qwen3TTSTestCheckpointPolicy() throws -> Qwen3TTSCheckpointTensorPolicy {
    try Qwen3TTSCheckpointTensorPolicy(
        cam: registryModuleIR("qwen3_tts")
    )
}

private func qwen3TTSTestTensorBlock(for name: String) -> String {
    if name.hasPrefix("decoder.") {
        return "codec-decoder"
    }
    if name.hasPrefix("talker.code_predictor.") {
        return "mtp-head"
    }
    if name.hasPrefix("talker.codec_head.") {
        return "codec-head"
    }
    if name == "talker.model.text_embedding.weight"
        || name.hasPrefix("talker.text_projection.") {
        return "tts-frontend"
    }
    if name.hasPrefix("talker.model.") {
        return "talker"
    }
    preconditionFailure("no Qwen3-TTS test tensor block for \(name)")
}
