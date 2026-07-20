// Qwen3TTSCodecGateTests — numerical gates for the Qwen3-TTS codec decoder against
// the real reference (framework gate-before-compose). Opt-in: set
//   SMELT_VOICE_MODEL  = path to the downloaded checkpoint dir
//   SMELT_VOICE_REFS   = dir of refs from tools/reference/extract_qwen_tts_refs.py
// When unset the gates no-op (the refs/weights aren't in CI). Bar: cosine >= 0.999.

import Foundation
import Testing
@testable import SmeltRuntime
import SmeltSchema

private func env(_ k: String) -> String? {
    guard let v = ProcessInfo.processInfo.environment[k], !v.isEmpty else { return nil }
    return v
}

private func readF32(_ path: String) -> [Float] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func readI32(_ path: String) -> [Int32] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
}

private func loadF32Tensor(_ loader: SafetensorsLoader, _ name: String) -> [Float] {
    let info = loader.tensor(named: name)!
    precondition(info.dtype == "F32", "\(name) is \(info.dtype), expected F32")
    let count = info.shape.reduce(1, *)
    let src = loader.tensorData(info)
    return Array(UnsafeBufferPointer(
        start: src.assumingMemoryBound(to: Float.self), count: count))
}

/// Load F32 or BF16 tensor as [Float] (BF16 upcast = bits << 16, matching torch's
/// fp32 load of the bf16 talker — the reference was computed that way).
private func loadTensor(_ loader: SafetensorsLoader, _ name: String) -> [Float] {
    let info = loader.tensor(named: name)!
    let count = info.shape.reduce(1, *)
    let src = loader.tensorData(info)
    switch info.dtype {
    case "F32":
        return Array(UnsafeBufferPointer(start: src.assumingMemoryBound(to: Float.self), count: count))
    case "BF16":
        let bf = UnsafeBufferPointer(start: src.assumingMemoryBound(to: UInt16.self), count: count)
        return bf.map { Float(bitPattern: UInt32($0) << 16) }
    default:
        fatalError("unsupported dtype \(info.dtype) for \(name)")
    }
}

/// Assert GPU/CPU output matches the reference by BOTH cosine (direction) and
/// relative L2 (magnitude) — cosine alone is blind to a global gain/offset error.
private func assertMatch(_ out: [Float], _ ref: [Float], _ label: String,
                         cosMin: Float = 0.999, relMax: Float = 0.02) {
    #expect(out.count == ref.count, "\(label): count \(out.count) vs \(ref.count)")
    let n = min(out.count, ref.count)
    var dot: Float = 0, na: Float = 0, nb: Float = 0, diff: Float = 0
    for i in 0..<n {
        dot += out[i] * ref[i]; na += out[i] * out[i]; nb += ref[i] * ref[i]
        let d = out[i] - ref[i]; diff += d * d
    }
    let cos = dot / (na.squareRoot() * nb.squareRoot())
    let relL2 = diff.squareRoot() / nb.squareRoot()
    #expect(cos >= cosMin, "\(label): cosine \(cos) < \(cosMin)")
    #expect(relL2 <= relMax, "\(label): relL2 \(relL2) > \(relMax) (gain/offset error)")
}

/// tts_pad_embed = text_projection(text_embedding(tts_pad_id)); tts_pad_id = prefill_ids[2].
private func loadTTSPadEmbed(_ loader: SafetensorsLoader, _ refs: String) -> [Float] {
    func ld(_ n: String) -> [Float] { loadTensor(loader, n) }
    let ttsPadId = readI32("\(refs)/prefill_ids.bin").map { Int($0) }[2]
    var padRow = [Float](repeating: 0, count: 2048)
    let textEmb = ld("talker.model.text_embedding.weight")
    for d in 0..<2048 { padRow[d] = textEmb[ttsPadId * 2048 + d] }
    return Qwen3TTSTalker.textProjection(
        padRow, rows: 1,
        fc1W: ld("talker.text_projection.linear_fc1.weight"), fc1B: ld("talker.text_projection.linear_fc1.bias"),
        fc2W: ld("talker.text_projection.linear_fc2.weight"), fc2B: ld("talker.text_projection.linear_fc2.bias"))
}

/// Load `count` Qwen3 decoder layers under `prefix` into Talker.Layer structs. Shared by the
/// talker (28L) and MTP (5L) gates — the layer weight layout is identical between them.
private func loadTalkerLayers(_ loader: SafetensorsLoader, prefix: String, count: Int) -> [Qwen3TTSTalker.Layer] {
    func t(_ n: String) -> [Float] { loadTensor(loader, "\(prefix)\(n)") }
    var layers: [Qwen3TTSTalker.Layer] = []
    for i in 0..<count {
        let p = "layers.\(i)."
        layers.append(.init(
            inputNorm: t("\(p)input_layernorm.weight"), postAttnNorm: t("\(p)post_attention_layernorm.weight"),
            qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
            vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
            qNorm: t("\(p)self_attn.q_norm.weight"), kNorm: t("\(p)self_attn.k_norm.weight"),
            gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"),
            downProj: t("\(p)mlp.down_proj.weight")))
    }
    return layers
}

@Test func qwen3TTSCodecRVQDequantMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return  // opt-in gate; refs/weights not present
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])

    let dim = 256
    let firstEmb = Qwen3TTSCodec.codebookEmbedding(
        embeddingSum: loadF32Tensor(loader, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
        clusterUsage: loadF32Tensor(loader, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"),
        dim: dim)
    var restEmb: [[Float]] = []
    for k in 0..<15 {
        restEmb.append(Qwen3TTSCodec.codebookEmbedding(
            embeddingSum: loadF32Tensor(loader, "decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.embedding_sum"),
            clusterUsage: loadF32Tensor(loader, "decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.cluster_usage"),
            dim: dim))
    }
    let firstProj = loadF32Tensor(loader, "decoder.quantizer.rvq_first.output_proj.weight")  // [512,256,1]
    let restProj = loadF32Tensor(loader, "decoder.quantizer.rvq_rest.output_proj.weight")

    // input_codes [1,16,T] C-contiguous -> [16][T].
    let codesFlat = readI32("\(refs)/input_codes.bin")
    let frames = codesFlat.count / 16
    var codesKT: [[Int32]] = []
    for k in 0..<16 { codesKT.append(Array(codesFlat[(k * frames)..<((k + 1) * frames)])) }

    let out = Qwen3TTSCodec.rvqDequantize(
        codesKT: codesKT, firstEmb: firstEmb, restEmb: restEmb,
        firstProj: firstProj, restProj: restProj)

    let ref = readF32("\(refs)/stage_quantizer.bin")  // [1,512,T] C-contiguous -> [512,T]
    assertMatch(out, ref, "RVQ dequant")
}

@Test func qwen3TTSCodecPreConvMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
    let weight = loadF32Tensor(loader, "decoder.pre_conv.conv.weight")  // [1024,512,3]
    let bias = loadF32Tensor(loader, "decoder.pre_conv.conv.bias")      // [1024]

    // Feed the reference RVQ output to isolate pre_conv. [1,512,T] -> [512,T].
    let input = readF32("\(refs)/stage_quantizer.bin")
    let lengthIn = input.count / 512
    let (out, lengthOut) = Qwen3TTSCodec.causalConv1d(
        input: input, inChannels: 512, lengthIn: lengthIn,
        weight: weight, bias: bias,
        outChannels: 1024, kernel: 3, stride: 1)

    let ref = readF32("\(refs)/stage_pre_conv.bin")  // [1,1024,T]
    #expect(lengthOut == lengthIn)
    assertMatch(out, ref, "pre_conv")
}

@Test func qwen3TTSCodecPreTransformerMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
    func t(_ name: String) -> [Float] { loadF32Tensor(loader, "decoder.pre_transformer.\(name)") }

    var layers: [Qwen3TTSCodec.PreTransformerLayer] = []
    for i in 0..<8 {
        let p = "layers.\(i)."
        layers.append(.init(
            inputNorm: t("\(p)input_layernorm.weight"),
            postAttnNorm: t("\(p)post_attention_layernorm.weight"),
            qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
            vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
            attnScale: t("\(p)self_attn_layer_scale.scale"), mlpScale: t("\(p)mlp_layer_scale.scale"),
            gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"),
            downProj: t("\(p)mlp.down_proj.weight")))
    }
    let w = Qwen3TTSCodec.PreTransformerWeights(
        inputProjW: t("input_proj.weight"), inputProjB: t("input_proj.bias"),
        outputProjW: t("output_proj.weight"), outputProjB: t("output_proj.bias"),
        normW: t("norm.weight"), layers: layers)

    // pre_transformer input = pre_conv output transposed [1,1024,T] -> [T,1024].
    let preconv = readF32("\(refs)/stage_pre_conv.bin")
    let frames = preconv.count / 1024
    var input = [Float](repeating: 0, count: frames * 1024)
    for c in 0..<1024 { for tt in 0..<frames { input[tt * 1024 + c] = preconv[c * frames + tt] } }

    let out = Qwen3TTSCodec.preTransformer(input: input, frames: frames, w: w)

    let ref = readF32("\(refs)/stage_pre_transformer.bin")  // [1,T,1024] -> row-major [T,1024]
    assertMatch(out, ref, "pre_transformer")
}

@Test func qwen3TTSCodecUpsampleMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
    func cn(_ b: Int) -> Qwen3TTSCodec.ConvNeXt {
        let p = "decoder.upsample.\(b).1."
        return .init(
            dwconvW: loadF32Tensor(loader, "\(p)dwconv.conv.weight"),
            dwconvB: loadF32Tensor(loader, "\(p)dwconv.conv.bias"),
            normW: loadF32Tensor(loader, "\(p)norm.weight"), normB: loadF32Tensor(loader, "\(p)norm.bias"),
            pw1W: loadF32Tensor(loader, "\(p)pwconv1.weight"), pw1B: loadF32Tensor(loader, "\(p)pwconv1.bias"),
            pw2W: loadF32Tensor(loader, "\(p)pwconv2.weight"), pw2B: loadF32Tensor(loader, "\(p)pwconv2.bias"),
            gamma: loadF32Tensor(loader, "\(p)gamma"))
    }
    var blocks: [Qwen3TTSCodec.UpsampleBlock] = []
    for b in 0..<2 {
        blocks.append(.init(
            transW: loadF32Tensor(loader, "decoder.upsample.\(b).0.conv.weight"),
            transB: loadF32Tensor(loader, "decoder.upsample.\(b).0.conv.bias"),
            convNext: cn(b)))
    }

    // input = pre_transformer output permuted to [1024, T].
    let pt = readF32("\(refs)/stage_pre_transformer.bin")  // [1,T,1024]
    let frames = pt.count / 1024
    var input = [Float](repeating: 0, count: 1024 * frames)
    for tt in 0..<frames { for c in 0..<1024 { input[c * frames + tt] = pt[tt * 1024 + c] } }

    let (out, lengthOut) = Qwen3TTSCodec.upsample(input: input, dim: 1024, lengthIn: frames, blocks: blocks)
    let ref = readF32("\(refs)/stage_post_upsample.bin")  // [1,1024,4T]
    #expect(lengthOut == frames * 4)
    assertMatch(out, ref, "upsample")
}

@Test func qwen3TTSCodecDecoderMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
    func ld(_ n: String) -> [Float] { loadF32Tensor(loader, n) }
    func snake(_ p: String) -> Qwen3TTSCodec.SnakeBetaW {
        .init(alpha: ld("\(p).alpha"), beta: ld("\(p).beta"))
    }

    let rates = [8, 5, 4, 3], dilations = [1, 3, 9]
    var blocks: [Qwen3TTSCodec.DecoderBlockW] = []
    for i in 0..<4 {
        let p = "decoder.decoder.\(i + 1).block"
        let inDim = 1536 >> i, outDim = 1536 >> (i + 1)
        var resUnits: [Qwen3TTSCodec.ResUnitW] = []
        for (ri, dil) in dilations.enumerated() {
            let rp = "\(p).\(ri + 2)"
            resUnits.append(.init(
                act1: snake("\(rp).act1"), conv1W: ld("\(rp).conv1.conv.weight"), conv1B: ld("\(rp).conv1.conv.bias"),
                act2: snake("\(rp).act2"), conv2W: ld("\(rp).conv2.conv.weight"), conv2B: ld("\(rp).conv2.conv.bias"),
                dilation: dil))
        }
        blocks.append(.init(
            snake: snake("\(p).0"), transW: ld("\(p).1.conv.weight"), transB: ld("\(p).1.conv.bias"),
            rate: rates[i], resUnits: resUnits, inDim: inDim, outDim: outDim))
    }
    let w = Qwen3TTSCodec.DecoderW(
        conv0W: ld("decoder.decoder.0.conv.weight"), conv0B: ld("decoder.decoder.0.conv.bias"),
        blocks: blocks, finalSnake: snake("decoder.decoder.5"),
        finalConvW: ld("decoder.decoder.6.conv.weight"), finalConvB: ld("decoder.decoder.6.conv.bias"))

    // input = reference post-upsample [1,1024,T].
    let pu = readF32("\(refs)/stage_post_upsample.bin")
    let frames = pu.count / 1024
    let out = Qwen3TTSCodec.decoder(input: pu, latent: 1024, frames: frames, w: w)

    let ref = readF32("\(refs)/stage_decoder_wav.bin")  // [1,1,T*480]
    assertMatch(out, ref, "decoder")
}

private func loadCodecWeights(_ loader: SafetensorsLoader) -> Qwen3TTSCodec.CodecWeights {
    func ld(_ n: String) -> [Float] { loadF32Tensor(loader, n) }
    func snake(_ p: String) -> Qwen3TTSCodec.SnakeBetaW { .init(alpha: ld("\(p).alpha"), beta: ld("\(p).beta")) }

    let firstEmb = Qwen3TTSCodec.codebookEmbedding(
        embeddingSum: ld("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
        clusterUsage: ld("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"), dim: 256)
    var restEmb: [[Float]] = []
    for k in 0..<15 {
        restEmb.append(Qwen3TTSCodec.codebookEmbedding(
            embeddingSum: ld("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.embedding_sum"),
            clusterUsage: ld("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.cluster_usage"), dim: 256))
    }

    var ptLayers: [Qwen3TTSCodec.PreTransformerLayer] = []
    for i in 0..<8 {
        let p = "decoder.pre_transformer.layers.\(i)."
        ptLayers.append(.init(
            inputNorm: ld("\(p)input_layernorm.weight"), postAttnNorm: ld("\(p)post_attention_layernorm.weight"),
            qProj: ld("\(p)self_attn.q_proj.weight"), kProj: ld("\(p)self_attn.k_proj.weight"),
            vProj: ld("\(p)self_attn.v_proj.weight"), oProj: ld("\(p)self_attn.o_proj.weight"),
            attnScale: ld("\(p)self_attn_layer_scale.scale"), mlpScale: ld("\(p)mlp_layer_scale.scale"),
            gateProj: ld("\(p)mlp.gate_proj.weight"), upProj: ld("\(p)mlp.up_proj.weight"),
            downProj: ld("\(p)mlp.down_proj.weight")))
    }
    let pt = Qwen3TTSCodec.PreTransformerWeights(
        inputProjW: ld("decoder.pre_transformer.input_proj.weight"), inputProjB: ld("decoder.pre_transformer.input_proj.bias"),
        outputProjW: ld("decoder.pre_transformer.output_proj.weight"), outputProjB: ld("decoder.pre_transformer.output_proj.bias"),
        normW: ld("decoder.pre_transformer.norm.weight"), layers: ptLayers)

    var upBlocks: [Qwen3TTSCodec.UpsampleBlock] = []
    for b in 0..<2 {
        let p = "decoder.upsample.\(b).1."
        upBlocks.append(.init(
            transW: ld("decoder.upsample.\(b).0.conv.weight"), transB: ld("decoder.upsample.\(b).0.conv.bias"),
            convNext: .init(
                dwconvW: ld("\(p)dwconv.conv.weight"), dwconvB: ld("\(p)dwconv.conv.bias"),
                normW: ld("\(p)norm.weight"), normB: ld("\(p)norm.bias"),
                pw1W: ld("\(p)pwconv1.weight"), pw1B: ld("\(p)pwconv1.bias"),
                pw2W: ld("\(p)pwconv2.weight"), pw2B: ld("\(p)pwconv2.bias"), gamma: ld("\(p)gamma"))))
    }

    let rates = [8, 5, 4, 3], dilations = [1, 3, 9]
    var blocks: [Qwen3TTSCodec.DecoderBlockW] = []
    for i in 0..<4 {
        let p = "decoder.decoder.\(i + 1).block"
        var resUnits: [Qwen3TTSCodec.ResUnitW] = []
        for (ri, dil) in dilations.enumerated() {
            let rp = "\(p).\(ri + 2)"
            resUnits.append(.init(
                act1: snake("\(rp).act1"), conv1W: ld("\(rp).conv1.conv.weight"), conv1B: ld("\(rp).conv1.conv.bias"),
                act2: snake("\(rp).act2"), conv2W: ld("\(rp).conv2.conv.weight"), conv2B: ld("\(rp).conv2.conv.bias"),
                dilation: dil))
        }
        blocks.append(.init(snake: snake("\(p).0"), transW: ld("\(p).1.conv.weight"), transB: ld("\(p).1.conv.bias"),
                            rate: rates[i], resUnits: resUnits, inDim: 1536 >> i, outDim: 1536 >> (i + 1)))
    }
    let dec = Qwen3TTSCodec.DecoderW(
        conv0W: ld("decoder.decoder.0.conv.weight"), conv0B: ld("decoder.decoder.0.conv.bias"), blocks: blocks,
        finalSnake: snake("decoder.decoder.5"), finalConvW: ld("decoder.decoder.6.conv.weight"),
        finalConvB: ld("decoder.decoder.6.conv.bias"))

    return .init(
        firstEmb: firstEmb, restEmb: restEmb,
        firstProj: ld("decoder.quantizer.rvq_first.output_proj.weight"),
        restProj: ld("decoder.quantizer.rvq_rest.output_proj.weight"),
        preConvW: ld("decoder.pre_conv.conv.weight"), preConvB: ld("decoder.pre_conv.conv.bias"),
        preTransformer: pt, upsampleBlocks: upBlocks, decoderW: dec)
}

@Test func qwen3TTSCodecEndToEndMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
    let w = loadCodecWeights(loader)

    let codesFlat = readI32("\(refs)/input_codes.bin")
    let frames = codesFlat.count / 16
    var codesKT: [[Int32]] = []
    for k in 0..<16 { codesKT.append(Array(codesFlat[(k * frames)..<((k + 1) * frames)])) }

    let out = Qwen3TTSCodec.decodeCodec(codesKT: codesKT, w: w)
    let ref = readF32("\(refs)/stage_decoder_wav.bin")  // [1,1,T*1920]
    assertMatch(out, ref, "end-to-end codec")
}

@Test func qwen3TTSTalkerForwardMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let w = Qwen3TTSTalker.Weights(
        normW: loadTensor(loader, "talker.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.model.", count: 28))

    // Guard the axis-0 shortcut: this gate uses cos/sin axis 0 only, valid ONLY if the
    // 3 mRoPE axes are identical (the extractor records this). Fail loudly otherwise.
    let meta = (try? String(contentsOfFile: "\(refs)/talker_meta.json", encoding: .utf8)) ?? ""
    #expect(meta.contains("\"mrope_axes_identical\": true"),
            "talker gate assumes mRoPE collapses to 1D RoPE; the ref capture disagrees")

    let ie = readF32("\(refs)/talker_inputs_embeds.bin")  // [1,T,2048] -> [T,2048]
    let frames = ie.count / 2048
    // cos/sin [3,1,T,128] C-contiguous; mRoPE axes identical -> use axis 0 = [T,128].
    // Coverage limits (codex): T=30 prefill cannot distinguish full-causal from a
    // sliding window >=30, and this gate does not exercise KV-cache decode steps or
    // the mRoPE derivation from position_ids — those are separate future gates.
    let cos = Array(readF32("\(refs)/talker_cos.bin")[0..<(frames * 128)])
    let sin = Array(readF32("\(refs)/talker_sin.bin")[0..<(frames * 128)])

    let out = Qwen3TTSTalker.forward(inputsEmbeds: ie, frames: frames, cos: cos, sin: sin, w: w)
    let ref = readF32("\(refs)/talker_last_hidden.bin")  // [1,T,2048]
    assertMatch(out, ref, "talker forward")
}

@Test func qwen3TTSTalkerCodecHeadMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let weight = loadTensor(loader, "talker.codec_head.weight")  // [3072,2048] BF16

    // Feed the reference last_hidden to isolate the head. [1,T,2048] -> [T,2048].
    let hidden = readF32("\(refs)/talker_last_hidden.bin")
    let frames = hidden.count / 2048
    let out = Qwen3TTSTalker.codecHead(hidden: hidden, frames: frames, weight: weight)

    let ref = readF32("\(refs)/talker_codec_logits.bin")  // [1,T,3072]
    assertMatch(out, ref, "talker codec_head")
}

@Test func qwen3TTSTalkerTextProjectionMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func t(_ n: String) -> [Float] { loadTensor(loader, "talker.text_projection.\(n)") }

    let input = readF32("\(refs)/text_proj_in.bin")  // [1,R,2048] -> [R,2048]
    let rows = input.count / 2048
    let out = Qwen3TTSTalker.textProjection(
        input, rows: rows,
        fc1W: t("linear_fc1.weight"), fc1B: t("linear_fc1.bias"),
        fc2W: t("linear_fc2.weight"), fc2B: t("linear_fc2.bias"))

    let ref = readF32("\(refs)/text_proj_out.bin")  // [1,R,2048]
    assertMatch(out, ref, "talker text_projection")
}

@Test func qwen3TTSTalkerRopeDerivationMatchesReference() throws {
    guard env("SMELT_VOICE_MODEL") != nil, let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    // rope_position_ids [3,1,T]: prefill positions, identical across the 3 mRoPE axes.
    let posAll = readF32("\(refs)/rope_position_ids.bin")
    let frames = posAll.count / 3
    let positions = Array(posAll[0..<frames])
    // Axes must be identical for the 1D-RoPE collapse the derivation assumes.
    for ax in 1..<3 {
        for t in 0..<frames {
            #expect(posAll[ax * frames + t] == positions[t], "mRoPE axis \(ax) differs at \(t)")
        }
    }

    let (cos, sin) = Qwen3TTSTalker.ropeCosSin(positions: positions)

    // talker_cos/sin [3,1,T,128]; axes identical -> compare axis 0 = [T,128].
    let refCos = Array(readF32("\(refs)/talker_cos.bin")[0..<(frames * 128)])
    let refSin = Array(readF32("\(refs)/talker_sin.bin")[0..<(frames * 128)])
    assertMatch(cos, refCos, "rope cos")
    assertMatch(sin, refSin, "rope sin")
}

@Test func qwen3TTSMTPProjectionMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let w = loadTensor(loader, "talker.code_predictor.small_to_mtp_projection.weight")  // [1024,2048]
    let b = loadTensor(loader, "talker.code_predictor.small_to_mtp_projection.bias")    // [1024]

    let input = readF32("\(refs)/mtp_proj_in.bin")  // [1,5,2048] -> [5,2048]
    let rows = input.count / 2048
    let out = Qwen3TTSMTP.projection(input, rows: rows, weight: w, bias: b)

    let ref = readF32("\(refs)/mtp_proj_out.bin")  // [1,5,1024]
    assertMatch(out, ref, "MTP small_to_mtp_projection")
}

@Test func qwen3TTSMTPTransformerMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let w = Qwen3TTSTalker.Weights(
        normW: loadTensor(loader, "talker.code_predictor.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.code_predictor.model.", count: 5))

    let ie = readF32("\(refs)/mtp_xf_in.bin")  // [1,16,1024] -> [16,1024]
    let frames = ie.count / 1024
    let out = Qwen3TTSMTP.transformer(inputsEmbeds: ie, frames: frames, w: w)

    let ref = readF32("\(refs)/mtp_xf_out.bin")  // [1,16,1024]
    assertMatch(out, ref, "MTP 5L transformer")
}

@Test func qwen3TTSMTPLmHeadMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let weight = loadTensor(loader, "talker.code_predictor.lm_head.0.weight")  // [2048,1024]

    let hidden = readF32("\(refs)/mtp_head_in.bin")  // [1,4,1024] -> [4,1024]
    let rows = hidden.count / 1024
    let out = Qwen3TTSMTP.lmHead(hidden, rows: rows, weight: weight)

    let ref = readF32("\(refs)/mtp_head0_out.bin")  // [1,4,2048]
    assertMatch(out, ref, "MTP lm_head[0]")
}

@Test func qwen3TTSMTPSubTalkerLogitsMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func ld(_ n: String) -> [Float] { loadTensor(loader, n) }

    let talkerCodecEmb = ld("talker.model.codec_embedding.weight")  // [3072,2048]
    var mtpCodecEmbs: [[Float]] = []
    for i in 0..<14 { mtpCodecEmbs.append(ld("talker.code_predictor.model.codec_embedding.\(i).weight")) }  // [2048,2048]
    var lmHeads: [[Float]] = []
    for i in 0..<15 { lmHeads.append(ld("talker.code_predictor.lm_head.\(i).weight")) }  // [2048,1024]
    let transformerW = Qwen3TTSTalker.Weights(
        normW: ld("talker.code_predictor.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.code_predictor.model.", count: 5))

    let talkerHidden = readF32("\(refs)/mtp_sub_talker_hidden.bin")          // [1,2048]
    let codecIds = readI32("\(refs)/mtp_sub_codec_ids.bin").map { Int($0) }  // [16]
    let out = Qwen3TTSMTP.subTalkerLogits(
        talkerHidden: talkerHidden, codecIds: codecIds,
        talkerCodecEmb: talkerCodecEmb, mtpCodecEmbs: mtpCodecEmbs,
        projW: ld("talker.code_predictor.small_to_mtp_projection.weight"),
        projB: ld("talker.code_predictor.small_to_mtp_projection.bias"),
        transformerW: transformerW, lmHeads: lmHeads)

    let ref = readF32("\(refs)/mtp_sub_logits.bin")  // [1,15,2048]
    assertMatch(out, ref, "MTP sub-talker logits (per-frame composition)")
}

// Front-end U1: the HF-slow Qwen tokenizer + chat-wrap reproduces the captured assembly id
// streams from the source strings alone (the only piece text→inputsEmbeds was missing).
@Test func qwen3TTSFrontEndChatWrapMatchesAssemblyIds() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let tok = try SmeltTokenizer(
        qwenVocabJSONPath: "\(model)/vocab.json",
        mergesTxtPath: "\(model)/merges.txt",
        tokenizerConfigPath: "\(model)/tokenizer_config.json")

    // Ground-truth source strings from tools/reference/extract_qwen_tts_talker_refs.py.
    let (instructIds, inputIds) = Qwen3TTSFrontEnd.wrap(
        text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.", tokenizer: tok)
    let refInstruct = readF32("\(refs)/assembly_instruct_id.bin").map { Int($0) }
    let refInput = readF32("\(refs)/assembly_input_id.bin").map { Int($0) }
    #expect(instructIds == refInstruct, "instruct ids \(instructIds) != ref \(refInstruct)")
    #expect(inputIds == refInput, "input ids \(inputIds) != ref \(refInput)")

    // Parity sweep vs HF (expected ids captured from Qwen2Tokenizer): locks the Qwen2 regex
    // (single-digit \p{N}), NFC, whitespace/CRLF/contraction handling, and special-aware split.
    #expect(tok.encode("12345").map(Int.init) == [16, 17, 18, 19, 20])
    #expect(tok.encode("a  b").map(Int.init) == [64, 220, 293])
    #expect(tok.encode("Hello\r\nWorld").map(Int.init) == [9707, 319, 10134])
    #expect(tok.encode("don't").map(Int.init) == [15007, 944])
    #expect(tok.encode(" leading").map(Int.init) == [6388])
    #expect(tok.encodeWithSpecials("x<|im_start|>y").map(Int.init) == [87, 151644, 88])
    #expect(tok.encode("#include").map(Int.init) == [1067])      // locks `#`-prefixed merges
    #expect(tok.encode("cafe\u{301}").map(Int.init) == [924, 58858])  // locks NFC (decomposed→é)
}

// Reproduces talker_inputs_embeds from the source strings ALONE (no captured ref ids — that is
// what distinguishes this from qwen3TTSTalkerVoiceDesignPrefillMatchesReference below).
@Test func qwen3TTSFrontEndTextToInputsEmbedsMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let tok = try SmeltTokenizer(
        qwenVocabJSONPath: "\(model)/vocab.json",
        mergesTxtPath: "\(model)/merges.txt",
        tokenizerConfigPath: "\(model)/tokenizer_config.json")
    let config = try Qwen3TTSFrontEnd.Config.load(configJSONPath: "\(model)/config.json")

    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func tp(_ n: String) -> [Float] { loadTensor(loader, "talker.text_projection.\(n)") }
    let (out, frames) = try Qwen3TTSFrontEnd.textToInputsEmbeds(
        text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.", language: "English",
        tokenizer: tok, config: config,
        textEmbedding: loadTensor(loader, "talker.model.text_embedding.weight"),
        fc1W: tp("linear_fc1.weight"), fc1B: tp("linear_fc1.bias"),
        fc2W: tp("linear_fc2.weight"), fc2B: tp("linear_fc2.bias"),
        codecEmbedding: loadTensor(loader, "talker.model.codec_embedding.weight"))

    let ref = readF32("\(refs)/talker_inputs_embeds.bin")
    #expect(frames == ref.count / 2048, "assembled \(frames) frames vs ref \(ref.count / 2048)")
    assertMatch(out, ref, "front-end text→inputsEmbeds")

    let p = readI32("\(refs)/prefill_ids.bin").map { Int($0) }
    let en = try config.ids(language: "english")
    #expect([en.ttsBos, en.ttsEos, en.ttsPad, en.codecThink, en.codecThinkBos,
             en.codecThinkEos, en.codecPad, en.codecBos, en.languageId] == p)
    #expect(try config.ids(language: "Spanish").languageId == 2054)
    #expect(throws: Qwen3TTSFrontEnd.Error.self) { _ = try config.ids(language: "klingon") }
}

// Qwen3TTSTalkerPrefill.layout is the shared structural spec the host voiceDesignPrefill and the GPU
// frontEndPrefillHiddenA both follow (one textId + optional codecId per output row). Pin its exact row
// order/content — instruct & role are projection-only (codecId nil); the think prefix is 4 rows of
// ttsPad+codec; nothink collapses to 3; the speaker row is present only for CustomVoice; bos/text/eos
// carry codecPad and the final row is ttsPad+codecBos. A drift here desyncs host vs GPU prefill.
@Test func qwen3TTSPrefillLayoutMatchesAssembly() {
    typealias Row = Qwen3TTSTalkerPrefill.PrefillRow
    // role[3] + text[3] + trailing[5] = 11; textLen = 3.
    let inputIds = [10, 11, 12, 200, 201, 202, 90, 91, 92, 93, 94]

    // Explicit language (think) + CustomVoice speaker.
    let think = Qwen3TTSTalkerPrefill.Ids(
        ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
        codecPad: 7, codecBos: 8, languageId: 50, codecNothink: nil, speakerId: 60)
    #expect(Qwen3TTSTalkerPrefill.layout(instructIds: [101, 102], inputIds: inputIds, ids: think) == [
        Row(textId: 101, codecId: nil), Row(textId: 102, codecId: nil),                 // instruct
        Row(textId: 10, codecId: nil), Row(textId: 11, codecId: nil), Row(textId: 12, codecId: nil), // role
        Row(textId: 3, codecId: 4), Row(textId: 3, codecId: 5),                          // think, think_bos
        Row(textId: 3, codecId: 50), Row(textId: 3, codecId: 6),                         // lang, think_eos
        Row(textId: 3, codecId: 60),                                                     // speaker
        Row(textId: 1, codecId: 7),                                                      // tts_bos
        Row(textId: 200, codecId: 7), Row(textId: 201, codecId: 7), Row(textId: 202, codecId: 7), // text
        Row(textId: 2, codecId: 7),                                                      // tts_eos
        Row(textId: 3, codecId: 8),                                                      // final
    ])

    // Auto language (nothink), no instruct, no speaker.
    let nothink = Qwen3TTSTalkerPrefill.Ids(
        ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
        codecPad: 7, codecBos: 8, languageId: nil, codecNothink: 40, speakerId: nil)
    #expect(Qwen3TTSTalkerPrefill.layout(instructIds: [], inputIds: inputIds, ids: nothink) == [
        Row(textId: 10, codecId: nil), Row(textId: 11, codecId: nil), Row(textId: 12, codecId: nil), // role
        Row(textId: 3, codecId: 40), Row(textId: 3, codecId: 5), Row(textId: 3, codecId: 6),         // nothink prefix
        Row(textId: 1, codecId: 7),                                                      // tts_bos
        Row(textId: 200, codecId: 7), Row(textId: 201, codecId: 7), Row(textId: 202, codecId: 7), // text
        Row(textId: 2, codecId: 7),                                                      // tts_eos
        Row(textId: 3, codecId: 8),                                                      // final
    ])
}

// layout() is the structural source of truth the GPU front-end depends on, but voiceDesignPrefill
// still encodes the order independently. This fast pure-CPU gate assembles from layout() (row-aligned
// gather → project → merge, using the SAME host helpers) and pins it equal to voiceDesignPrefill on
// synthetic weights — so a fast-CI edit to either order desyncs HERE, not only in the skipped
// checkpoint gate. Tolerance (not bit-exact): voiceDesignPrefill projects in 4 groups while this
// projects all rows at once, and cblas reduction can differ in the last bits; a layout drift diverges
// grossly, far above this bound.
@Test func qwen3TTSPrefillLayoutDrivenAssemblyMatchesVoiceDesignPrefill() {
    let dim = 8, inter = 6, hidden = 4, tableRows = 64
    func v(_ i: Int) -> Float { Float((i &* 2654435761) % 1000) / 1000.0 - 0.5 }   // deterministic in [-0.5, 0.5)
    let textEmbedding = (0..<tableRows * dim).map { v($0) }
    let codecEmbedding = (0..<tableRows * hidden).map { v($0 &+ 7) }
    let fc1W = (0..<inter * dim).map { v($0 &+ 11) }, fc1B = (0..<inter).map { v($0 &+ 13) }
    let fc2W = (0..<hidden * inter).map { v($0 &+ 17) }, fc2B = (0..<hidden).map { v($0 &+ 19) }
    let inputIds = [21, 22, 23, 31, 32, 33, 90, 91, 92, 93, 94]   // role[3] + text[3] + trailing[5]

    let think = Qwen3TTSTalkerPrefill.Ids(
        ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
        codecPad: 7, codecBos: 8, languageId: 50, codecNothink: nil, speakerId: 60)
    let nothink = Qwen3TTSTalkerPrefill.Ids(
        ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
        codecPad: 7, codecBos: 8, languageId: nil, codecNothink: 40, speakerId: nil)

    for (instructIds, ids) in [([11, 12], think), ([], nothink)] {
        let ref = Qwen3TTSTalkerPrefill.voiceDesignPrefill(
            instructIds: instructIds, inputIds: inputIds, ids: ids,
            textEmbedding: textEmbedding, fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B,
            codecEmbedding: codecEmbedding, dim: dim, projInter: inter, hidden: hidden).embeds

        // Layout-driven assembly with the same host primitives.
        let rows = Qwen3TTSTalkerPrefill.layout(instructIds: instructIds, inputIds: inputIds, ids: ids)
        let gathered = rows.flatMap { Array(textEmbedding[($0.textId * dim)..<(($0.textId + 1) * dim)]) }
        let projected = Qwen3TTSTalker.textProjection(
            gathered, rows: rows.count, fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B,
            inDim: dim, interDim: inter, outDim: hidden)
        var got = [Float](repeating: 0, count: rows.count * hidden)
        for (r, row) in rows.enumerated() {
            for d in 0..<hidden {
                let c = row.codecId.map { codecEmbedding[$0 * hidden + d] } ?? 0
                got[r * hidden + d] = projected[r * hidden + d] + c
            }
        }
        #expect(got.count == ref.count)
        let maxd = zip(got, ref).map { abs($0 - $1) }.max() ?? .infinity
        #expect(maxd < 1e-3, "layout-driven assembly vs voiceDesignPrefill maxAbsDiff \(maxd)")
    }
}

// Qwen3TTSFrontEndEmitter picks the matmul kernel by K%4, mirroring matmulDispatch: K%4==0 → gemm_tn_f32,
// else matmul_f32. The byte-identical checkpoint gate only hits gemm_tn_f32 (real Qwen dims are %4==0), so
// pin the matmul_f32 branch here on a synthetic K%4!=0 plan (codex 1a-ii review). Pipeline indices:
// gatherF32=0, gatherBF16=1, gemmTN=2, matmul=3, silu=4, srtc=5.
@Test func qwen3TTSFrontEndEmitterSelectsMatmulByK() {
    // textEmbedDim=6 (fc1 K=6, %4!=0 → matmul_f32); projInter=8 (fc2 K=8, %4==0 → gemm_tn_f32).
    let plan = Qwen3TTSFrontEndEmitter.plan(
        seqLen: 9, textEmbedDim: 6, projInter: 8, hidden: 4, textEmbedBF16: true)
    // records: [gatherText(bf16), fc1, silu, fc2, gatherCodec(f32), srtc]
    #expect(plan.records.count == 6)
    #expect(plan.records[0].pipeline == 1)                          // text gather → bf16
    #expect(plan.records[1].pipeline == 3)                          // fc1 K=6 %4!=0 → matmul_f32
    #expect(plan.records[1].gridW == 8 && plan.records[1].gridH == 9)   // matmul_f32 grid (N=projInter, M=seqLen)
    #expect(plan.records[2].pipeline == 4)                          // silu
    #expect(plan.records[3].pipeline == 2)                          // fc2 K=8 %4==0 → gemm_tn_f32
    #expect(plan.records[4].pipeline == 0)                          // codec gather → f32
    #expect(plan.records[5].pipeline == 5)                          // merge → scale_residual_tc

    // All-%4==0 dims keep both projections on gemm_tn_f32 (the production path).
    let tn = Qwen3TTSFrontEndEmitter.plan(
        seqLen: 9, textEmbedDim: 8, projInter: 8, hidden: 4, textEmbedBF16: true)
    #expect(tn.records[1].pipeline == 2 && tn.records[3].pipeline == 2)
}

@Test func qwen3TTSTalkerVoiceDesignPrefillMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func tp(_ n: String) -> [Float] { loadTensor(loader, "talker.text_projection.\(n)") }

    let instructIds = readF32("\(refs)/assembly_instruct_id.bin").map { Int($0) }
    let inputIds = readF32("\(refs)/assembly_input_id.bin").map { Int($0) }
    // Special ids from the captured model config (prefill_ids.bin), in extractor order.
    let p = readI32("\(refs)/prefill_ids.bin").map { Int($0) }
    let ids = Qwen3TTSTalkerPrefill.Ids(
        ttsBos: p[0], ttsEos: p[1], ttsPad: p[2],
        codecThink: p[3], codecThinkBos: p[4], codecThinkEos: p[5],
        codecPad: p[6], codecBos: p[7], languageId: p[8])

    let (out, frames) = Qwen3TTSTalkerPrefill.voiceDesignPrefill(
        instructIds: instructIds, inputIds: inputIds, ids: ids,
        textEmbedding: loadTensor(loader, "talker.model.text_embedding.weight"),
        fc1W: tp("linear_fc1.weight"), fc1B: tp("linear_fc1.bias"),
        fc2W: tp("linear_fc2.weight"), fc2B: tp("linear_fc2.bias"),
        codecEmbedding: loadTensor(loader, "talker.model.codec_embedding.weight"))

    let ref = readF32("\(refs)/talker_inputs_embeds.bin")  // [1,30,2048]
    #expect(frames == ref.count / 2048, "assembled \(frames) frames vs ref \(ref.count / 2048)")
    assertMatch(out, ref, "talker VoiceDesign dual-track prefill")
}

@Test func qwen3TTSTalkerDecodeForwardMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let w = Qwen3TTSTalker.Weights(
        normW: loadTensor(loader, "talker.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.model.", count: 28))
    let headW = loadTensor(loader, "talker.codec_head.weight")

    // Full greedy sequence inputs_embeds: prefill [0..29] ++ decode last-position rows
    // [30..55]. The gen capture's row 0 is prefill pos 29 (== prefill[29]), so skip it.
    // prefill and the gen-run prefill must match (same text/instruct; sampling-independent).
    let prefill = readF32("\(refs)/talker_inputs_embeds.bin")                 // [30,2048]
    let genIE = readF32("\(refs)/talker_gen_inputs_embeds.bin")               // [27,2048] pos 29..55
    let prefillLen = prefill.count / 2048
    let codes = readI32("\(refs)/gen_codes.bin")                              // [16,26] row-major
    let nFrames = codes.count / 16
    #expect(prefillLen == 30, "unexpected prefill length \(prefillLen)")
    #expect(genIE.count / 2048 == nFrames + 1, "gen rows \(genIE.count / 2048) vs frames+eos \(nFrames + 1)")
    let ie = prefill + Array(genIE[(1 * 2048)...])                            // pos 0..55
    let frames = ie.count / 2048

    let (cos, sin) = Qwen3TTSTalker.ropeCosSin(positions: (0..<frames).map(Float.init))
    let hidden = Qwen3TTSTalker.forward(inputsEmbeds: ie, frames: frames, cos: cos, sin: sin, w: w)

    // codecHead at the decode positions [29..55] -> logits per generated frame (+eos).
    let firstLogitPos = prefillLen - 1
    let decodeHidden = Array(hidden[(firstLogitPos * 2048)...])
    let nLogits = frames - firstLogitPos
    let out = Qwen3TTSTalker.codecHead(hidden: decodeHidden, frames: nLogits, weight: headW)

    let ref = readF32("\(refs)/talker_gen_logits.bin")  // [27,3072]
    #expect(nLogits == ref.count / 3072, "logits \(nLogits) vs ref \(ref.count / 3072)")
    assertMatch(out, ref, "talker decode-step codecHead logits")

    // Strongest check: greedy argmax of each frame's logits == the generated codebook-0
    // (codes row 0 = first nFrames elements of the row-major [16,nFrames] buffer).
    for k in 0..<nFrames {
        var am = 0, mx = -Float.greatestFiniteMagnitude
        for v in 0..<3072 { let x = out[k * 3072 + v]; if x > mx { mx = x; am = v } }
        #expect(am == Int(codes[k]), "frame \(k) argmax \(am) != generated code \(codes[k])")
    }
}

@Test func qwen3TTSTalkerKVCacheMatchesFullRecompute() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    let w = Qwen3TTSTalker.Weights(
        normW: loadTensor(loader, "talker.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.model.", count: 28))

    let prefill = readF32("\(refs)/talker_inputs_embeds.bin")     // [30,2048]
    let genIE = readF32("\(refs)/talker_gen_inputs_embeds.bin")   // [27,2048], rows 1.. = decode inputs
    let prefillLen = prefill.count / 2048
    let nSteps = genIE.count / 2048 - 1
    let ie = prefill + Array(genIE[(1 * 2048)...])                // full [prefillLen+nSteps, 2048]
    let frames = ie.count / 2048

    // Reference: full-recompute forward over the whole sequence.
    let (cos, sin) = Qwen3TTSTalker.ropeCosSin(positions: (0..<frames).map(Float.init))
    let full = Qwen3TTSTalker.forward(inputsEmbeds: ie, frames: frames, cos: cos, sin: sin, w: w)

    // Cached: prefill once, then one single-position step per frame.
    var cache = Qwen3TTSTalker.KVCache(layers: 28)
    var cached = Qwen3TTSTalker.forwardStep(
        newEmbeds: prefill, newFrames: prefillLen, startPos: 0, cache: &cache, w: w)
    for s in 0..<nSteps {
        let row = Array(genIE[((s + 1) * 2048)..<((s + 2) * 2048)])
        cached += Qwen3TTSTalker.forwardStep(
            newEmbeds: row, newFrames: 1, startPos: prefillLen + s, cache: &cache, w: w)
    }
    #expect(cached.count == full.count, "cached \(cached.count) vs full \(full.count)")
    assertMatch(cached, full, "talker KV-cache vs full recompute")
}

@Test func qwen3TTSNextFrameInputMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func ld(_ n: String) -> [Float] { loadTensor(loader, n) }

    let talkerCodecEmb = ld("talker.model.codec_embedding.weight")  // [3072,2048]
    var mtpCodecEmbs: [[Float]] = []
    for i in 0..<15 { mtpCodecEmbs.append(ld("talker.code_predictor.model.codec_embedding.\(i).weight")) }
    let ttsPadEmbed = loadTTSPadEmbed(loader, refs)

    // codes [16,nFrames] row-major; column k = the 16 codebooks of frame k.
    let codesFlat = readI32("\(refs)/gen_codes.bin").map { Int($0) }
    let nFrames = codesFlat.count / 16
    let genIE = readF32("\(refs)/talker_gen_inputs_embeds.bin")  // [nFrames+1,2048], rows 1.. = decode inputs

    // Frame k's 16 codes feed the decode input at genIE row k+1 (the last frame's next
    // input is consumed by the eos step). Check all nFrames against rows 1..nFrames.
    var assembled: [Float] = []
    for k in 0..<nFrames {
        let codes16 = (0..<16).map { codesFlat[$0 * nFrames + k] }
        assembled += Qwen3TTSGenerator.nextFrameInput(
            codes16: codes16, talkerCodecEmb: talkerCodecEmb,
            mtpCodecEmbs: mtpCodecEmbs, ttsPadEmbed: ttsPadEmbed)
    }
    let ref = Array(genIE[(1 * 2048)..<((nFrames + 1) * 2048)])  // rows 1..nFrames
    assertMatch(assembled, ref, "AR next-frame dual-track input")
}

@Test func qwen3TTSCb0ProcessorsMatchFullGreedy() throws {
    guard env("SMELT_VOICE_MODEL") != nil, let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    // Apply the cb0 logits processors (suppress + repetition_penalty + min_new_tokens) to
    // every captured raw codecHead logit row and confirm argmax reproduces the full greedy
    // run: codes[0,k] for the 26 frames, then codec_eos (2150) for the 27th (stop) row.
    let logits = readF32("\(refs)/talker_gen_logits.bin")  // [27,3072]
    let nRows = logits.count / 3072
    let codes = readI32("\(refs)/gen_codes.bin").map { Int($0) }
    let nFrames = codes.count / 16
    #expect(nRows == nFrames + 1, "rows \(nRows) vs frames+eos \(nFrames + 1)")

    var history: [Int] = []
    for k in 0..<nRows {
        let raw = Array(logits[(k * 3072)..<((k + 1) * 3072)])
        let processed = Qwen3TTSGenerator.applyCb0Processors(raw, history: history, frame: k)
        var am = 0, mx = -Float.greatestFiniteMagnitude
        for v in 0..<3072 where processed[v] > mx { mx = processed[v]; am = v }
        if k < nFrames {
            #expect(am == codes[k], "frame \(k): processed argmax \(am) != code \(codes[k])")
            history.append(codes[k])
        } else {
            #expect(am == 2150, "stop row: processed argmax \(am) != codec_eos 2150")
        }
    }
}

@Test func qwen3TTSGenerateGreedyMatchesReference() throws {
    guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
        return
    }
    let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
    func ld(_ n: String) -> [Float] { loadTensor(loader, n) }

    let talkerW = Qwen3TTSTalker.Weights(
        normW: ld("talker.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.model.", count: 28))
    let mtpW = Qwen3TTSTalker.Weights(
        normW: ld("talker.code_predictor.model.norm.weight"),
        layers: loadTalkerLayers(loader, prefix: "talker.code_predictor.model.", count: 5))
    var lmHeads: [[Float]] = [], mtpEmbs: [[Float]] = []
    for i in 0..<15 { lmHeads.append(ld("talker.code_predictor.lm_head.\(i).weight")) }
    for i in 0..<15 { mtpEmbs.append(ld("talker.code_predictor.model.codec_embedding.\(i).weight")) }
    let ttsPadEmbed = loadTTSPadEmbed(loader, refs)

    let w = Qwen3TTSGenerator.Weights(
        talker: talkerW, codecHeadW: ld("talker.codec_head.weight"),
        talkerCodecEmb: ld("talker.model.codec_embedding.weight"),
        mtp: mtpW, mtpProjW: ld("talker.code_predictor.small_to_mtp_projection.weight"),
        mtpProjB: ld("talker.code_predictor.small_to_mtp_projection.bias"),
        mtpLmHeads: lmHeads, mtpCodecEmbs: mtpEmbs, ttsPadEmbed: ttsPadEmbed,
        codecEosId: 2150)  // codec_eos_token_id (unused at this frame count; loop stops on maxFrames)

    let prefill = readF32("\(refs)/talker_inputs_embeds.bin")
    let prefillLen = prefill.count / 2048
    let codes = readI32("\(refs)/gen_codes.bin").map { Int($0) }
    let nFrames = codes.count / 16

    // Full free-running generation: with the KV cache each frame is O(T), so generate the
    // whole utterance and check it stops naturally on codec_eos at exactly nFrames frames.
    let gen = Qwen3TTSGenerator.generate(prefill: prefill, prefillLen: prefillLen, w: w, maxFrames: nFrames + 8)
    // gen is codebook-major [16][producedFrames], matching the row-major [16,nFrames] reference.
    #expect(gen.count == 16 && gen[0].count == nFrames,
            "produced \(gen.first?.count ?? -1) frames, expected \(nFrames) (natural eos stop)")
    for c in 0..<16 {
        for k in 0..<nFrames {
            #expect(gen[c][k] == codes[c * nFrames + k],
                    "cb \(c) frame \(k): generated \(gen[c][k]) != reference \(codes[c * nFrames + k])")
        }
    }
}
