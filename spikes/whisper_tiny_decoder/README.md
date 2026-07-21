# Whisper Tiny Decoder Spike

Standalone `whisper-tiny` decoder proof that does not depend on Smelt's main package.

Current scope:
- load `tools/whisper_ref/whisper-tiny.safetensors`
- load reference tensors from `tools/whisper_ref/dumps/passing-medium`
- run CPU greedy decode for the Whisper decoder only
- verify token parity and alignment-weight parity against the existing reference dump

This spike is intentionally narrow. It exists to answer one question first:

`Can we reproduce Whisper decoder behavior cleanly enough that replacing WhisperKit's decoder is worth it?`

## Run

```bash
cd spikes/whisper_tiny_decoder
swift test
```

The package auto-discovers the repo root by walking upward until it finds `tools/whisper_ref`.
When that optional fixture tree is absent, the package still builds and its
reference-dependent tests skip cleanly.

To typecheck the optional WhisperKit bridge against a local WhisperKit checkout:

```bash
cd spikes/whisper_tiny_decoder
WHISPERKIT_PATH=/path/to/whisperkit swift build --target WhisperTinyDecoderSpike
```

## Current Runtime Boundary

`WhisperTinyCPUDecoder` is shaped like the future runtime:
- input: encoder output `[1500, 384]`
- setup: precompute cross-attention K/V once per layer
- loop: token embedding + positional embedding -> 4 decoder layers -> final LayerNorm -> LM head
- output: greedy token sequence and per-step alignment rows

`decodeStep(token:state:)` is the stable integration seam. `greedyDecode(...)` now just builds on top of that step API, which is the thing we want to keep when swapping the CPU internals for Metal later.

## WhisperKit Bridge

When `WHISPERKIT_PATH` is provided, the package also builds:
- `WhisperTinyDecodingInputs`
- `WhisperTinyWhisperKitDecoder`

These live in `Sources/WhisperTinyDecoderSpike/WhisperKitBridge.swift` and provide a thin `TextDecoding` adapter around the step runtime.

Current bridge limitations:
- no CoreML-style KV cache interchange through `predictLogits`
- no prefill-cache model support
- alignment weights are returned, but compute still runs on CPU
- intended for proving integration shape, not for shipping performance

Expected WhisperKit-side loader patch:
- `WhisperKit.loadModels()` should only require `TextDecoder` / `AudioEncoder` / `MelSpectrogram` bundles when the corresponding component is a `WhisperMLModel`
- if a custom `textDecoder` is already supplied and it is not a `WhisperMLModel`, `loadModels()` should skip the `TextDecoder` bundle existence check

In practice that means the current hard check over `[logmelUrl, encoderUrl, decoderUrl]` should become conditional per component instead of unconditional.

## Tiny Runtime Buffers For The Metal Follow-Up

Decoder constants:
- `d_model = 384`
- `layers = 4`
- `heads = 6`
- `head_dim = 64`
- `ffn_dim = 1536`
- `src_len = 1500`
- `tgt_len = 448`
- `vocab = 51865`

Proposed FP16 runtime buffers:
- `encoder_output` `[1500, 384]`
- `cross_k[layer]` `[1500, 384]`
- `cross_v[layer]` `[1500, 384]`
- `self_k[layer]` `[448, 384]`
- `self_v[layer]` `[448, 384]`
- `hidden` `[384]`
- `normed` `[384]`
- `q` `[384]`
- `k_step` `[384]`
- `v_step` `[384]`
- `attn_out` `[384]`
- `ffn_1` `[1536]`
- `ffn_2` `[384]`
- `logits` `[51865]`
- `alignment_row` `[1500]`

## Kernel Plan After CPU Parity

The first Metal pass should add only these kernels:
- `token_embedding_plus_position`
- `layer_norm_bias`
- `linear_bias_fp16`
- `self_attention_decode`
- `cross_attention_decode_alignment`
- `gelu`
- `elementwise_add`
- `argmax_fp16_or_fp32`

Everything else can stay hardcoded for `whisper-tiny` until parity and timing look good.
