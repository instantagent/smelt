// SmeltKernelCatalog — Maps every Metal kernel to its function name and binding counts.
//
// The catalog enables compile-time validation that generated dispatch code
// binds the correct number of buffers and constants to each kernel.
// Pipeline rawValues match the architecture plan order and the manifest's
// `pipelines` array index.

// MARK: - Pipeline enum

/// One case per Metal compute kernel, rawValue = pipeline index in manifest order.
public enum SmeltPipeline: Int, CaseIterable, Sendable {
    case fusedLutMatvec = 0
    case rmsNorm1PW = 1
    case rmsNormGated = 2
    case conv1dUpdateSilu = 3
    case l2Normalize = 4
    case computeGates = 5
    case stateDecay = 6
    case kvMemReadout = 7
    case computeDelta = 8
    case outerProductUpdate = 9
    case queryReadout = 10
    case swigluFused = 11
    case elementwiseAdd = 12
    case embeddingGather = 13
    case argmaxFP16 = 14
    case applyRope = 15
    case kvCacheUpdate = 16
    case attentionDecode = 17
    case sigmoidKernel = 18
    case elementwiseMul = 19
    case bufferCopy = 20
    case fp16Matvec = 21
    case gateSplit = 22
    case perHeadRmsNorm1PW = 23
    case lutEmbeddingGather = 24
    case kvReadoutDelta = 31
    case stateDecayUpdate = 32
    case affineMatvec = 33
    case fusedLutMatvecAdd = 34
    case sigmoidMul = 35
    case gatesKvReadoutDelta = 36
    case fusedGateUpSwiglu = 37
    case fusedDualLutMatvec = 38
    case fusedAffineMatvecAdd = 39
    case deltanetRecurrenceFusedDecode = 40
    case fusedDualAffineMatvec = 41
    case fusedAffineGateUpSwiglu = 42
    case fusedRmsNormAffineGateUpSwiglu = 43
    case fusedRmsNormAffineMatvec = 44
    case rmsNormScaleOnly = 45
    case normScaleAffineGateUpSwiglu = 46
    case normScaleAffineMatvec = 47
    case atomicNormAffineGateUpSwiglu = 48
    case atomicNormAffineMatvec = 49
    case deltanetRecurrenceMlxDecode = 50
    case conv1dUpdateSilu6144x4 = 51
    case l2NormalizeD128 = 52
    case rmsNormGatedD128 = 53
    case deltanetRecurrenceMlxDecodeD128H16 = 54
    case attentionDecodeD256H8KV2 = 55
    case rmsNorm1PWD2048 = 56
    case fusedAffineGateUpSwigluC2048R6144G64 = 57
    case fusedAffineMatvecAddC2048R2048G64 = 58
    case fusedAffineMatvecAddC6144R2048G64 = 59
    case affineMatvecC2048R2048G64 = 60
    case affineMatvecC2048R6144G64 = 61
    case affineMatvecC2048R4096G64 = 62
    case affineMatvecC2048R512G64 = 63
    case affineEmbeddingGather = 64
    case fusedDualAffineMatvecC2048R16G64Batched = 65
    case fusedAffineGateUpSwigluC2048R6144G64Batched = 66
    case affineMatvecC2048R2048G64Batched = 67
    case affineMatvecC2048R6144G64Batched = 68
    case affineMatvecC2048R4096G64Batched = 69
    case affineMatvecC2048R512G64Batched = 70
    case affineMatvecC6144R2048G64Batched = 71
    case fusedDualAffineMatvecC2048R512G64Batched = 72
    case conv1dUpdateSilu6144x4Prefill = 73
    case l2NormalizeQD128C6144H16Prefill = 74
    case l2NormalizeKD128C6144H16Prefill = 75
    case deltanetRecurrenceMlxPrefillD128H16 = 76
    case fusedAffineGateUpSwigluC2048R6144G64BatchedFull = 77
    case affineMatvecC2048R2048G64BatchedFull = 78
    case affineMatvecC2048R6144G64BatchedFull = 79
    case affineMatvecC2048R4096G64BatchedFull = 80
    case affineMatvecC2048R512G64BatchedFull = 81
    case affineMatvecC6144R2048G64BatchedFull = 82
    case affineEmbeddingGatherBatched = 83
    case attentionDecodeD256H8KV2SDPA = 84
    case perHeadRmsNorm1PWBatched = 85
    case affineMatvecC6144R2048G64 = 86
    case affineMatvecC2048R248320G64 = 87
    case fusedDualAffineMatvecC2048R16G64 = 88
    case conv1dUpdateSiluPrefill = 89
    case l2NormalizeQPrefill = 90
    case l2NormalizeKPrefill = 91
    case deltanetRecurrenceMlxPrefill = 92
    case rmsNorm1PWD1024 = 93
    case fusedAffineGateUpSwigluC1024R3584G64 = 94
    case affineMatvecC1024R2048G64 = 95
    case affineMatvecC1024R3584G64 = 96
    case affineMatvecC1024R4096G64 = 97
    case affineMatvecC1024R512G64 = 98
    case affineMatvecC1024R6144G64 = 99
    case affineMatvecC2048R1024G64 = 100
    case affineMatvecC1024R248320G64 = 101
    case fusedDualAffineMatvecC1024R16G64 = 102
    case fusedDualAffineMatvecC1024R512G64Batched = 103
    case fusedDualAffineMatvecC1024R16G64Batched = 104
    case fusedAffineGateUpSwigluC1024R3584G64BatchedFull = 105
    case affineMatvecC1024R2048G64BatchedFull = 106
    case affineMatvecC1024R3584G64BatchedFull = 107
    case affineMatvecC1024R4096G64BatchedFull = 108
    case affineMatvecC1024R512G64BatchedFull = 109
    case affineMatvecC1024R6144G64BatchedFull = 110
    case affineMatvecC2048R1024G64BatchedFull = 111
    case rmsNorm1PWD2560 = 112
    case fusedAffineGateUpSwigluC2560R9216G64 = 113
    case affineMatvecC2560R8192G64 = 114
    case affineMatvecC2560R4096G64 = 115
    case affineMatvecC2560R1024G64 = 116
    case affineMatvecC4096R2560G64 = 117
    case affineMatvecC9216R2560G64 = 118
    case affineMatvecC2560R248320G64 = 119
    case fusedDualAffineMatvecC2560R32G64 = 120
    case fusedDualAffineMatvecC2560R1024G64Batched = 121
    case fusedDualAffineMatvecC2560R32G64Batched = 122
    case fusedAffineGateUpSwigluC2560R9216G64BatchedFull = 123
    case affineMatvecC2560R8192G64BatchedFull = 124
    case affineMatvecC2560R4096G64BatchedFull = 125
    case affineMatvecC2560R1024G64BatchedFull = 126
    case affineMatvecC4096R2560G64BatchedFull = 127
    case affineMatvecC9216R2560G64BatchedFull = 128
    case affineMatvecC3584R1024G64 = 129
    case affineMatvecC3584R1024G64BatchedFull = 130
    case rmsNormGatedD128Batched = 131
    case rmsNorm1PWD1024Batched = 132
    case deltanetRecurrenceMlxDecodeD128H32QK16 = 133
    case deltanetRecurrenceMlxPrefillD128H32QK16 = 134
    case attentionDecodeD256H16KV4 = 135
    case attentionDecodeD256H16KV4SDPA = 136
    case affineMatvecC1024R6144G64Rows4 = 137
    case affineMatvecC2048R1024G64Rows4 = 138
    case affineMatvecC3584R1024G64Rows4 = 139
    case affineMatvecC1024R2048G64Rows4 = 140
    case affineMatvecC1024R248320G64Rows4 = 141
    case fusedAffineGateUpSwigluC1024R3584G64Rows4 = 142
    case gegluFused = 143
    case logitCap = 144
    case perHeadRmsNorm = 145
    case perHeadRmsNormNoScale = 146
    case attentionDecodeSoftcap = 147
    case scalarMul = 148
    case perHeadRmsNormBatched = 149
    case perHeadRmsNormNoScaleBatched = 150
    case attentionPrefillSoftcap = 151
    case fp16MatvecFP32Out = 152
    case rmsNorm1PWFromFP32 = 153
    case scalarMulWeight = 154
    case rmsNorm1PWFromFP32Batched = 155
    case rmsNorm1PWD1536 = 156
    case rmsNorm1PWD1536Batched = 157
    case affineMatvecC1536R2048G128 = 158
    case affineMatvecC1536R4096G128 = 159
    case affineMatvecC1536R256G128 = 160
    case affineMatvecC1536R512G128 = 161
    case affineMatvecC1536R6144G128 = 162
    case affineMatvecC1536R12288G128 = 163
    case affineMatvecC2048R1536G128 = 164
    case affineMatvecC4096R1536G128 = 165
    case affineMatvecC6144R1536G128 = 166
    case affineMatvecC12288R1536G128 = 167
    case fusedDualAffineMatvecC1536R256G128Batched = 168
    case fusedDualAffineMatvecC1536R512G128Batched = 169
    case affineMatvecC1536R2048G128BatchedFull = 170
    case affineMatvecC1536R4096G128BatchedFull = 171
    case affineMatvecC1536R6144G128BatchedFull = 172
    case affineMatvecC1536R12288G128BatchedFull = 173
    case affineMatvecC2048R1536G128BatchedFull = 174
    case affineMatvecC4096R1536G128BatchedFull = 175
    case affineMatvecC6144R1536G128BatchedFull = 176
    case affineMatvecC12288R1536G128BatchedFull = 177
    case affineMatvecC1536R2048G128Batched = 178
    case affineMatvecC1536R4096G128Batched = 179
    case affineMatvecC1536R6144G128Batched = 180
    case affineMatvecC1536R12288G128Batched = 181
    case affineMatvecC2048R1536G128Batched = 182
    case affineMatvecC4096R1536G128Batched = 183
    case affineMatvecC6144R1536G128Batched = 184
    case affineMatvecC12288R1536G128Batched = 185
    case attentionDecodeD256H8KV1 = 186
    case attentionDecodeD256H8KV1SDPA = 187
    case attentionDecodeD512H8KV1 = 188
    case attentionDecodeD512H8KV1SDPA = 189
    case affineMatvecC1536R2048G128Rows4 = 190
    case affineMatvecC1536R6144G128Rows4 = 191
    case affineMatvecC1536R12288G128Rows4 = 192
    case affineMatvecC2048R1536G128Rows4 = 193
    case affineMatvecC6144R1536G128Rows4 = 194
    case affineMatvecC12288R1536G128Rows4 = 195
    case affineMatvecC1536R262144G128Rows4 = 196
    case normScaleAffineMatvecC1536R2048G128Rows4 = 197
    case normScaleAffineMatvecC1536R12288G128Rows4 = 198
    case normScaleAffineMatvecC1536R262144G128Rows4 = 199
    case affineMatvecC1536R256G128Rows4 = 200
    case affineMatvecC256R1536G128Rows4 = 201
    case rmsNorm1PWD1536Add = 202
    case fusedAffineGateUpGeGLU = 203
    case fusedAffineGateUpGeGLUC1536R6144G128Rows4 = 204
    case fusedAffineGateUpGeGLUC1536R12288G128Rows4 = 205
    case fusedAffineMatvecAddC2048R1536G128Rows4 = 206
    case fusedAffineMatvecAddC4096R1536G128 = 207
    case fusedAffineMatvecAddC6144R1536G128Rows4 = 208
    case fusedAffineMatvecAddC12288R1536G128Rows4 = 209
    case rmsNormScaleOnlyD1536 = 210
    case fusedAffineGateUpGeGLUC1536R6144G128Rows8 = 211
    case fusedAffineGateUpGeGLUC1536R12288G128Rows8 = 212
    case affineMatvecC1536R256G128Rows8 = 213
    case normScaleAffineGateUpGeGLUC1536R6144G128Rows4 = 214
    case normScaleAffineGateUpGeGLUC1536R12288G128Rows4 = 215
    case affineMatvecC1536R262144G128Rows8 = 216
    case normScaleAffineMatvecC1536R262144G128Rows8 = 217
    case attentionDecodeD256H8KV1Fused = 218
    case attentionDecodeD256H8KV1FusedSoftcap = 219
    case attentionDecodeD256H8KV1FusedShared = 220
    case attentionDecodeD256H8KV1FusedSharedSoftcap = 221
    case attentionDecodeD512H8KV1Fused = 222
    case attentionDecodeD512H8KV1FusedSoftcap = 223
    case attentionDecodeD512H8KV1FusedShared = 224
    case attentionDecodeD512H8KV1FusedSharedSoftcap = 225
    case fusedAffineGateUpSwigluC2048R8192G64BatchedFull = 226
    case affineMatvecC8192R2048G64BatchedFull = 227
    case rmsNorm1PWD1536AddBatched = 228
    case fusedAffineMatvecAddC2048R1024G64BatchedFull = 229
    case fusedAffineMatvecAddC3584R1024G64BatchedFull = 230
    case fusedAffineMatvecAddC2048R2048G64BatchedFull = 231
    case fusedAffineMatvecAddC6144R2048G64BatchedFull = 232
    case fusedAffineMatvecAddC8192R2048G64BatchedFull = 233
    case fusedAffineMatvecAddC4096R2560G64BatchedFull = 234
    case fusedAffineMatvecAddC9216R2560G64BatchedFull = 235
    case rmsNormScaleOnlyD1024Batched = 236
    case rmsNormScaleOnlyD2048Batched = 237
    case rmsNormScaleOnlyD2560Batched = 238
    case normScaleAffineMatvecC1024R6144G64BatchedFull = 239
    case normScaleAffineMatvecC1024R4096G64BatchedFull = 240
    case normScaleAffineMatvecC2048R2048G64BatchedFull = 241
    case normScaleAffineMatvecC2048R6144G64BatchedFull = 242
    case normScaleAffineMatvecC2048R4096G64BatchedFull = 243
    case normScaleAffineMatvecC2560R8192G64BatchedFull = 244
    case normScaleAffineGateUpSwigluC1024R3584G64BatchedFull = 245
    case normScaleAffineGateUpSwigluC2048R6144G64BatchedFull = 246
    case normScaleAffineGateUpSwigluC2048R8192G64BatchedFull = 247
    case normScaleAffineGateUpSwigluC2560R9216G64BatchedFull = 248
    case rmsNormScaleOnlyD2048Eps1e5Batched = 249
    case fusedAffineGateUpSwigluC3072R8192G64 = 250
    case affineMatvecC3072R3072G64 = 251
    case affineMatvecC3072R1024G64 = 252
    case affineMatvecC3072R8192G64 = 253
    case affineMatvecC8192R3072G64 = 254
    case fusedDualAffineMatvecC3072R1024G64Batched = 255
    case fusedAffineGateUpSwigluC3072R8192G64BatchedFull = 256
    case affineMatvecC3072R3072G64BatchedFull = 257
    case affineMatvecC3072R1024G64BatchedFull = 258
    case affineMatvecC8192R3072G64BatchedFull = 259
    case fusedAffineMatvecAddC3072R3072G64BatchedFull = 260
    case fusedAffineMatvecAddC8192R3072G64BatchedFull = 261
    case normScaleAffineMatvecC3072R3072G64BatchedFull = 262
    case normScaleAffineGateUpSwigluC3072R8192G64BatchedFull = 263
    case rmsNormScaleOnlyD3072Eps1e5Batched = 264
    case rmsNorm1PWD2560Add = 265
    case rmsNorm1PWD2560Batched = 266
    case affineMatvecC2560R2048G128Rows4 = 267
    case affineMatvecC2560R4096G128Rows4 = 268
    case affineMatvecC2560R512G128Rows4 = 269
    case affineMatvecC2560R1024G128Rows4 = 270
    case affineMatvecC2560R10240G128Rows4 = 271
    case affineMatvecC2048R2560G128Rows4 = 272
    case affineMatvecC4096R2560G128Rows4 = 273
    case affineMatvecC10240R2560G128Rows4 = 274
    case affineMatvecC2560R256G128Rows4 = 275
    case affineMatvecC256R2560G128Rows4 = 276
    case affineMatvecC2560R262144G128Rows8 = 277
    case fusedAffineGateUpGeGLUC2560R10240G128Rows4 = 278
    case fusedDualAffineMatvecC2560R512G128Batched = 279
    case fusedDualAffineMatvecC2560R1024G128Batched = 280
    case affineMatvecC2560R2048G128Batched = 281
    case affineMatvecC2560R4096G128Batched = 282
    case affineMatvecC2560R512G128Batched = 283
    case affineMatvecC2560R1024G128Batched = 284
    case affineMatvecC2560R10240G128Batched = 285
    case affineMatvecC2048R2560G128Batched = 286
    case affineMatvecC4096R2560G128Batched = 287
    case affineMatvecC10240R2560G128Batched = 288
    case affineMatvecC2560R256G128Batched = 289
    case affineMatvecC256R2560G128Batched = 290
    case rmsNorm1PWD2560AddBatched = 291
    case affineMatvecC2560R10752G128Rows4 = 292
    case affineMatvecC2560R10752G128Batched = 293
    case rmsNormScaleOnlyD2560 = 294
    case normScaleAffineGateUpGeGLUC2560R10240G128Rows4 = 295
    case normAddScaleAffineMatvecC2560R256G128Rows4 = 296
    case fusedAffineGateUpGeGLUC2560R10240G128Batched = 297
    case normScaleAffineMatvecC2560R10240G128Batched = 298
    case fusedNormRopeAndKvCachePrefill = 299
    case affineMatvecC10240R2560G128BatchedTile4 = 300
    case fusedAffineGateUpGeGLUC2560R10240G128BatchedFull = 301
    /// Reserved slots (rawValues 302/303). `SmeltPipeline` uses explicit
    /// rawValues and `SmeltKernelCatalog.signatures` is a positional array
    /// indexed by `pipeline.rawValue`, so a removed middle case must leave a
    /// placeholder or every later pipeline's signature misaligns. These two
    /// held a dead attention-decode variant; no shipped model emits them.
    case attentionDecodeD256H8KV2SDPAReserved = 302
    case attentionDecodeD512H8KV2SDPAReserved = 303
    case affineMatvecC10240R2560G128Rows4SG1 = 304
    /// Sparse cluster lm_head: top-k centroid selection +
    /// scoped lm_head gather + dot-product + sparse scatter into the
    /// vocab-sized logits buffer. Inputs:
    ///   - centroidLogitsBuf [num_centroids] fp16  (centroid scores)
    ///   - lmHeadWeight       [vocab, hidden] fp16   (must NOT be
    ///     quantized; v1 kernel binds a single weight buffer with no
    ///     LUT/scales/biases slot. validateSmeltIR rejects
    ///     `cluster_embedder` + `quantize_embedding true` — the
    ///     ~78 MB embedder fits in fp16 without needing u4
    ///     compression, and supporting quantized cluster lm_head
    ///     would mean threading 3 more buffers through the kernel
    ///     signature. Future unit can extend.)
    ///   - hiddenStateBuf    [hidden]                  (last hidden)
    ///   - tokenOrderingBuf  [vocab] int32           (cluster permutation)
    ///   - logitsBuf         [vocab] fp16              (output, scattered)
    /// Constants: num_centroids, top_k, vocab_size, hidden_size,
    ///             tokens_per_cluster, logit_cap (fp32; 0 = disabled).
    /// Metal source: Resources/Shaders/cluster.metal.
    case clusterSparseLMHead = 305
    case fusedAffineGateUpGeGLUC256R2048G128Rows4 = 306
    case normScaleAffineGateUpGeGLUC256R2048G128Rows4 = 307
    case affineMatvecC256R1024G128Rows4 = 308
    case affineMatvecC256R2048G128Rows4 = 309
    case affineMatvecC1024R256G128Rows4 = 310
    case affineMatvecC2048R256G128Rows4 = 311
    case attentionDecodeD256H4KV2QnormRopeShared = 312
    case attentionDecodeD512H4KV2QnormRopeShared = 313
    case normScaleAffineGateUpGeGLUC2560R10240G128BatchedFull = 314
    case normScaleAffineMatvecC2560R2048G128Batched = 315
    case normScaleAffineMatvecC2560R4096G128Batched = 316
    case normScaleFusedDualAffineMatvecC2560R512G128Batched = 317
    case normScaleFusedDualAffineMatvecC2560R1024G128Batched = 318
    case affineMatvecArgmaxC1536R262144G128Batched = 319
    case affineMatvecArgmaxC2560R262144G128Batched = 320
    case lmHeadArgmaxReduceR262144 = 321
    case affineMatvecC2560R2048G128BatchedSG4BT5 = 322
    case gegluFusedStridedBatched = 323
    case affineMatvecC2048R2560G128BatchedSG4BT5 = 324
    case affineMatvecC10240R2560G128BatchedSG4BT5 = 325
    case affineMatvecC2560R10240G128BatchedSG4BT5 = 326
    case rmsNorm1PWD256Add = 327
    case rmsNorm1PWD256AddScalarWeight = 328
    case normScaleAffineMatvecC256R1024G128Rows4 = 329
    case normScaleAffineMatvecC256R2048G128Rows4 = 330
    case affineMatvecC2560R10240G128BatchedExtB5 = 331
    case affineMatvecC10240R2560G128BatchedExtB5 = 332
    case tqhEmbeddingGather = 333
    case tqhMatvecPrepareInput = 334
    case tqhMatvec = 335
    case tqhMatvecPrepareInputBatched = 336
    case tqhMatvecBatched = 337
    case affineMatvecC2560R2048G128BatchedExtB5 = 338
    case affineMatvecC2048R2560G128BatchedExtB5 = 339
    case affineMatvecC2560R10240G128BatchedExtB4 = 340
    case affineMatvecC10240R2560G128BatchedExtB4 = 341
    case affineMatvecC2560R2048G128BatchedExtB4 = 342
    case affineMatvecC2048R2560G128BatchedExtB4 = 343
    case fusedAffineGateUpGeGLUC2560R10240G128BatchedBT4SG4 = 344
    case affineMatvecC2560R262144G128Batched = 345
    case affineMatvecC10240R2560G128BatchedTile3 = 346
    case affineMatvecC2560R2048G128BatchedTile3 = 347
    case affineMatvecC2048R2560G128BatchedTile3 = 348
    case affineMatvecC2560R4096G128BatchedTile3 = 349
    case affineMatvecC4096R2560G128BatchedTile3 = 350

    // --- TTS kernels ---
    case snakeActivation = 351
    case convTranspose1d = 352
    case layerNorm = 353
    case conv1dForward = 354

    // --- Qwen3-TTS fp32 GPU kernels (codec + talker + MTP; gated == real model) ---
    case snakeBetaF32 = 355
    case conv1dForwardF32 = 356
    case convTranspose1dF32 = 357
    case layerNormCTF32 = 358
    case matmulF32 = 359
    case geluF32 = 360
    case siluF32 = 361
    case swigluF32 = 362
    case rmsNormCodecF32 = 363
    case rmsNormHeadF32 = 364
    case ropeApplyF32 = 365
    case slidingAttnF32 = 366
    case causalGQAAttnF32 = 367
    case rvqGatherSumF32 = 368
    case scaleResidualF32 = 369
    case scaleResidualTCF32 = 370
    case clampF32 = 371
    case decodeGQAAttnF32 = 372
    case matmulF16WF32 = 373
    case gemvF32 = 374
    case gemvF16WF32 = 375
    case argmaxF32 = 376
    case gatherRowF32 = 377
    case gemmF32 = 378
    case gemmF16WF32 = 379
    case channelCopyF32 = 380
    case nextFrameInputF32 = 381
    case cb0ArgmaxF32 = 382
    case gemvBF16WF32 = 383
    case gemmBF16WF32 = 384
    case sampleTopKF32 = 385
    case gemvU4F32 = 386
    case gemmU4F32 = 387
    case gatherRowBF16WF32 = 388
    case nextFrameInputBF16WF32 = 389
    case gemmTNF32 = 390
    case gemmTNF16WF32 = 391
    case gemmTNBF16WF32 = 392
    case causalGQAAttnSimdF32 = 393
    case slidingAttnSimdF32 = 394
    case gemvQKVBF16WF32 = 395
    case gemvGateUpSwigluBF16WF32 = 396
    case headNormRopeF32 = 397
    case gemvAddBF16WF32 = 398
    case causalGQAAttnCachedF32 = 399
    case transposeF32 = 400
    case cb0SampleTopKF32 = 401
    case gatherRowsF32 = 402
    case gatherRowsBF16WF32 = 403
    case causalGQAAttnCachedScalarF32 = 404
    case fp16MatvecBF16W = 405          // fp16-act × bf16-weight dense matvec (U2)
    case fp16MatvecFP32W = 406          // fp16-act × fp32-weight dense matvec (U2)
    case projectionBiasAddBatched = 407 // prefill: out[b,row] = input[b,row] + bias[row]
    case attentionDecodeD128H16KV2SDPA = 408
    case fusedAffineGateUpSwigluC2048R11008G64 = 409
    case argmaxFP16Partials = 410
    case argmaxKeyReduce = 411
    case affineMatvecC2048R151936G64Rows8 = 412
    case ropeKVCacheUpdate = 413
    case fusedAffineMatvecAddC2048R1024G64Rows4 = 414
    case fusedDualAffineMatvecC1024R16G64Rows4 = 415
    case conv1dUpdateSiluL2QKC6144K4D128H16 = 416
    case conv1dUpdateSiluPrefillCheckpoint = 417
    case deltanetRecurrenceMlxPrefillCheckpoint = 418
    case signedBinaryMatvecG128Rows8 = 419
    case signedTernaryMatvecG128Rows8 = 420
    case signedBinaryEmbeddingGatherG128 = 421
    case signedTernaryEmbeddingGatherG128 = 422
    case signedBinaryGateUpSwigluG128Rows8 = 423
    case signedBinaryMatvecAddG128Rows8 = 424
    case attentionDecodeD256H24KV4 = 425
    case signedBinaryMatvecG128Rows8BatchedB4 = 426
    case signedTernaryMatvecG128Rows8BatchedB4 = 427
    case signedBinaryGateUpSwigluG128Rows8BatchedB4 = 428
    case signedBinaryPackedBank4MatvecG128Rows8 = 429
    case signedActivationBitplanesI3G128 = 430
    case signedBinaryBitplaneI3Bank4MatvecG128Rows8 = 431
    case signedActivationBitplanesI2G128 = 432
    case signedBinaryBitplaneI2Bank4MatvecG128Rows8 = 433
    case signedBinaryBitplaneI3MatvecG128Rows8 = 434
    case signedBinaryBitplaneI2MatvecG128Rows8 = 435
    case signedActivationBitplanesI4G128 = 436
    case signedBinaryBitplaneI4MatvecG128Rows8 = 437
    case signedBinaryBitplaneI4MatvecAddG128Rows8 = 438
    case signedBinaryBitplaneI3MatvecAddG128Rows8 = 439
    case signedBinaryBitplaneI2MatvecAddG128Rows8 = 440
    case normScaleSignedActivationBitplanesI4G128 = 441
    case normScaleSignedActivationBitplanesI3G128 = 442
    case normScaleSignedActivationBitplanesI2G128 = 443
    case signedActivationBitplanesI5G128 = 444
    case signedBinaryBitplaneI5MatvecG128Rows8 = 445
    case signedActivationBitplanesI6G128 = 446
    case signedBinaryBitplaneI6MatvecG128Rows8 = 447
    case signedBinaryBitplaneI5MatvecAddG128Rows8 = 448
    case signedBinaryBitplaneI6MatvecAddG128Rows8 = 449
    case rmsNormGatedD128SignedActivationBitplanesI6G128 = 450
    case sigmoidMulSignedActivationBitplanesI6G128 = 451
    case signedBinaryBitplaneI3GateUpSwigluG128Rows8 = 452
    case signedBinaryBitplaneI4GateUpSwigluG128Rows8 = 453
    case signedBinaryBitplaneI5GateUpSwigluG128Rows8 = 454
    case signedBinaryBitplaneI6GateUpSwigluG128Rows8 = 455
    case rmsScaleQK = 456
    case signedTernaryBitplaneI4MatvecG128Rows8 = 457
    case signedTernaryBitplaneI5MatvecG128Rows8 = 458
    case signedTernaryBitplaneI6MatvecG128Rows8 = 459
    case swigluSignedActivationBitplanesI5G128 = 460
    case normScaleSignedActivationBitplanesI5G128 = 461
    case normScaleSignedActivationBitplanesI6G128 = 462
    case rmsNormScaleOnlyPrecise = 463
    case signedTernaryBitplaneI4Bank4MatvecG128Rows8 = 464
    case signedTernaryBitplaneI5Bank4MatvecG128Rows8 = 465
    case signedTernaryBitplaneI6Bank4MatvecG128Rows8 = 466
    case signedTernaryBitplaneI4GateUpSwigluG128Rows8 = 467
    case signedTernaryBitplaneI5GateUpSwigluG128Rows8 = 468
    case signedTernaryBitplaneI6GateUpSwigluG128Rows8 = 469
    case signedTernaryBitplaneI4MatvecG128Rows2Wide = 470
    case signedTernaryBitplaneI5MatvecG128Rows2Wide = 471
    case signedTernaryBitplaneI6MatvecG128Rows2Wide = 472
    case residualAddRMSNormScaleOnlyPrecise = 473
    case deltanetRecurrenceMlxDecodeD128H48QK16 = 474
    case signedActivationBitplanesI2G128Batched = 475
    case signedActivationBitplanesI3G128Batched = 476
    case signedActivationBitplanesI4G128Batched = 477
    case signedActivationBitplanesI5G128Batched = 478
    case signedActivationBitplanesI6G128Batched = 479
    case signedBinaryBitplaneI2MatvecG128Rows8BatchedB4 = 480
    case signedBinaryBitplaneI3MatvecG128Rows8BatchedB4 = 481
    case signedBinaryBitplaneI4MatvecG128Rows8BatchedB4 = 482
    case signedBinaryBitplaneI5MatvecG128Rows8BatchedB4 = 483
    case signedBinaryBitplaneI6MatvecG128Rows8BatchedB4 = 484
    case signedBinaryBitplaneI2MatvecG128Rows8BatchedB8 = 485
    case signedBinaryBitplaneI3MatvecG128Rows8BatchedB8 = 486
    case signedBinaryBitplaneI4MatvecG128Rows8BatchedB8 = 487
    case signedBinaryBitplaneI5MatvecG128Rows8BatchedB8 = 488
    case signedBinaryBitplaneI6MatvecG128Rows8BatchedB8 = 489
    case deltanetRecurrenceMlxPrefillD128H48QK16 = 490
    case attentionDecodeD256H24KV4SDPA = 491
    case signedTernaryAffineMatvecG128Rows8 = 492
    case ropeAndKvCachePrefillAnalytic = 493
    case attentionPrefillSDPAVectorD256 = 494
    case signedTernaryAffineQMMG128BM32BN32BK32 = 495
    case attentionPrefillMLXFallbackD256 = 496
    case attentionDecodeMLXVectorD256 = 497
    case attentionDecodeMLXVector2Pass1D256B128 = 498
    case attentionDecodeMLXVector2Pass2D256B128 = 499
    case signedTernaryAffineGateUpSwigluG128Rows8 = 500
    case signedTernaryAffineMatvecAddG128Rows8 = 501
    case signedTernaryAffineBank4MatvecG128Rows8 = 502
    case noncausalAttentionF32 = 503
    case layerNormRowsF32 = 504
    case fourierPositionEmbeddingF32 = 505
    case fsqBase8x5DecodeF32 = 506
    case addRowsF32 = 507
    case sigmoidF32 = 508
    case denseBF16WF32 = 509
    case appendStridedFeaturesF32 = 510
    case layerNormRowsBF16WF32 = 511
    case extractInterleavedHeadPartF32 = 512
    case rmsNormRowsBF16WF32 = 513
    case repackConcatenatedHeadPartsF32 = 514
    case rmsNormCodecBF16WF32 = 515
    case rmsNormHeadBF16WF32 = 516
    case headNormRopeBF16WF32 = 517
    case pmpeBF16SemanticsF32 = 518
    case rmsNormCodecBF16 = 519
    case rmsNormHeadBF16 = 520
    case headNormRopeBF16 = 521
    case gemvQKVBF16 = 522
    case gemvAddBF16 = 523
    case gemvGateUpSwigluBF16 = 524
    case decodeGQAAttnBF16 = 525
    case gemmBF16 = 526
    case ropeApplyBF16 = 527
    case causalGQAAttnCachedBF16 = 528
    case scaleResidualTCBF16 = 529
    case swigluBF16 = 530
    case gatherRowBF16 = 531
    case denseBF16 = 532
    case noncausalAttentionUpdateF32 = 533
    case denseBF16WF32Rows4 = 534
    case denseBF16WF32Rows8 = 535
    case denseBF16WF32Rows8Epilogue = 536
    case noncausalAttentionQ8F32 = 537

    // --- Prefill batched kernels ---
    case fusedLutMatmul = 25
    case rmsNorm1PWBatched = 26
    case embeddingGatherBatched = 27
    case attentionPrefill = 28
    case deltanetRecurrencePrefill = 29
    case ropeAndKvCachePrefill = 30
}

// MARK: - Kernel signature

/// Describes one Metal kernel's expected binding layout.
public struct SmeltKernelSignature: Sendable {
    /// Which pipeline this signature belongs to.
    public let pipeline: SmeltPipeline
    /// The Metal function name (must match the `kernel void` name in .metal source).
    public let metalFunctionName: String
    /// Number of `device` buffer bindings (setBuffer calls).
    public let bufferBindingCount: Int
    /// Number of `constant` bindings (setBytes calls).
    public let constantCount: Int

    public init(
        pipeline: SmeltPipeline,
        metalFunctionName: String,
        bufferBindingCount: Int,
        constantCount: Int
    ) {
        self.pipeline = pipeline
        self.metalFunctionName = metalFunctionName
        self.bufferBindingCount = bufferBindingCount
        self.constantCount = constantCount
    }
}

// MARK: - Catalog

/// Static catalog of all kernel signatures, indexed by pipeline rawValue.
public enum SmeltKernelCatalog {

    /// All kernel signatures in pipeline-index order.
    /// `signatures[pipeline.rawValue]` gives the signature for that pipeline.
    public static let signatures: [SmeltKernelSignature] = [
        // 0: fusedLutMatvec — indices, lut, input, output + numRows
        //    cols and groupSize are now function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedLutMatvec,
            metalFunctionName: "fused_lut_matvec",
            bufferBindingCount: 4,
            constantCount: 1
        ),
        // 1: rmsNorm1PW — input, weight, output + dim, eps
        SmeltKernelSignature(
            pipeline: .rmsNorm1PW,
            metalFunctionName: "rms_norm_1pw",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 2: rmsNormGated — input, gate, weight, output + headDim, eps
        SmeltKernelSignature(
            pipeline: .rmsNormGated,
            metalFunctionName: "rms_norm_gated",
            bufferBindingCount: 4,
            constantCount: 2
        ),
        // 3: conv1dUpdateSilu — state, new_val, conv_weight, output + channels, kernelSize
        SmeltKernelSignature(
            pipeline: .conv1dUpdateSilu,
            metalFunctionName: "conv1d_update_silu",
            bufferBindingCount: 4,
            constantCount: 2
        ),
        // 4: l2Normalize — data (in-place) + headDim, eps
        SmeltKernelSignature(
            pipeline: .l2Normalize,
            metalFunctionName: "l2_normalize",
            bufferBindingCount: 1,
            constantCount: 2
        ),
        // 5: computeGates — b_proj, a_proj, a_log, dt_bias, beta_out, g_out + numHeads
        SmeltKernelSignature(
            pipeline: .computeGates,
            metalFunctionName: "compute_gates",
            bufferBindingCount: 6,
            constantCount: 1
        ),
        // 6: stateDecay — state, g_val + headDim
        SmeltKernelSignature(
            pipeline: .stateDecay,
            metalFunctionName: "state_decay",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        // 7: kvMemReadout — state, key, kv_mem + headDim
        SmeltKernelSignature(
            pipeline: .kvMemReadout,
            metalFunctionName: "kv_mem_readout",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 8: computeDelta — value, kv_mem, beta, delta + headDim
        SmeltKernelSignature(
            pipeline: .computeDelta,
            metalFunctionName: "compute_delta",
            bufferBindingCount: 4,
            constantCount: 1
        ),
        // 9: outerProductUpdate — state, key, delta + headDim
        SmeltKernelSignature(
            pipeline: .outerProductUpdate,
            metalFunctionName: "outer_product_update",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 10: queryReadout — state, query, output + headDim, scale
        SmeltKernelSignature(
            pipeline: .queryReadout,
            metalFunctionName: "query_readout",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 11: swigluFused — gate, up, output + count
        SmeltKernelSignature(
            pipeline: .swigluFused,
            metalFunctionName: "swiglu_fused",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 12: elementwiseAdd — inputA, inputB, output + count
        SmeltKernelSignature(
            pipeline: .elementwiseAdd,
            metalFunctionName: "elementwise_add",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 13: embeddingGather — table, index, output + hidden
        SmeltKernelSignature(
            pipeline: .embeddingGather,
            metalFunctionName: "embedding_gather",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 14: argmaxFP16 — logits, result + vocabSize
        SmeltKernelSignature(
            pipeline: .argmaxFP16,
            metalFunctionName: "argmax_fp16",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        // 15: applyRope — qk, cos_val, sin_val
        //     + headDim, ropeDim, numHeads, layout, position, log2(base), math mode
        SmeltKernelSignature(
            pipeline: .applyRope,
            metalFunctionName: "apply_rope",
            bufferBindingCount: 3,
            constantCount: 7
        ),
        // 16: kvCacheUpdate — cache, new_kv + maxSeq, headDim, pos, numHeads
        SmeltKernelSignature(
            pipeline: .kvCacheUpdate,
            metalFunctionName: "kv_cache_update",
            bufferBindingCount: 2,
            constantCount: 4
        ),
        // 17: attentionDecode — query, k_cache, v_cache, attn_mask, output
        //     + headDim, maxSeq, seqLen, numKVHeads, scale, slidingWindow
        SmeltKernelSignature(
            pipeline: .attentionDecode,
            metalFunctionName: "attention_decode",
            bufferBindingCount: 5,
            constantCount: 6
        ),
        // 18: sigmoidKernel — input, output + count
        SmeltKernelSignature(
            pipeline: .sigmoidKernel,
            metalFunctionName: "sigmoid_kernel",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        // 19: elementwiseMul — inputA, inputB, output + count
        SmeltKernelSignature(
            pipeline: .elementwiseMul,
            metalFunctionName: "elementwise_mul",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 20: bufferCopy — src, dst + count
        SmeltKernelSignature(
            pipeline: .bufferCopy,
            metalFunctionName: "buffer_copy",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        // 21: fp16Matvec — weight, input, output + cols
        SmeltKernelSignature(
            pipeline: .fp16Matvec,
            metalFunctionName: "fp16_matvec",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 22: gateSplit — input, query, gate + nHeads, headDim
        SmeltKernelSignature(
            pipeline: .gateSplit,
            metalFunctionName: "gate_split",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 23: perHeadRmsNorm1PW — input, weight, output + headDim, eps
        SmeltKernelSignature(
            pipeline: .perHeadRmsNorm1PW,
            metalFunctionName: "per_head_rms_norm_1pw",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 24: lutEmbeddingGather — indices, lut, tokenId, output + cols, groupSize
        SmeltKernelSignature(
            pipeline: .lutEmbeddingGather,
            metalFunctionName: "lut_embedding_gather",
            bufferBindingCount: 4,
            constantCount: 2
        ),

        // --- Prefill batched kernels ---

        // 25: fusedLutMatmul — indices, lut, input[B,C], output[B,R] + cols, groupSize, numRows
        SmeltKernelSignature(
            pipeline: .fusedLutMatmul,
            metalFunctionName: "fused_lut_matmul",
            bufferBindingCount: 4,
            constantCount: 3
        ),
        // 26: rmsNorm1PWBatched — input[B,dim], weight, output[B,dim] + dim, eps
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWBatched,
            metalFunctionName: "rms_norm_1pw_batched",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 27: embeddingGatherBatched — table, indices[B], output[B,hidden] + hidden, batchSize
        SmeltKernelSignature(
            pipeline: .embeddingGatherBatched,
            metalFunctionName: "embedding_gather_batched",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 28: attentionPrefill — Q[B,qH,D], K_cache[kvH,S,D], V_cache[kvH,S,D], out[B,qH,D]
        //     + headDim, seqLen, startPos, cacheSeqCapacity, numKVHeads, scale, slidingWindow
        SmeltKernelSignature(
            pipeline: .attentionPrefill,
            metalFunctionName: "attention_prefill",
            bufferBindingCount: 4,
            constantCount: 7
        ),
        // 29: deltanetRecurrencePrefill — fused conv+norms+gates+recurrence for all positions
        //     state, convState, qkvBuf, convWeight, aLog, dtBias, bBuf, aBuf, recOut
        //     + headDim, numHeads, seqLen, qkvDim, convKernel, headScale, l2Eps
        SmeltKernelSignature(
            pipeline: .deltanetRecurrencePrefill,
            metalFunctionName: "deltanet_recurrence_prefill",
            bufferBindingCount: 9,
            constantCount: 7
        ),
        // 30: ropeAndKvCachePrefill — batched RoPE + KV cache write
        //     queries, keys, values, cos_table, sin_table, key_cache, val_cache
        //     + headDim, ropeDim, qHeads, kvHeads, seqLen, startPos, cacheSeqCapacity, layout
        SmeltKernelSignature(
            pipeline: .ropeAndKvCachePrefill,
            metalFunctionName: "rope_and_kv_cache_prefill",
            bufferBindingCount: 7,
            constantCount: 8
        ),
        // 31: kvReadoutDelta — fused kv_mem readout + delta computation
        //     state, key, value, g_val, beta, delta + headDim
        SmeltKernelSignature(
            pipeline: .kvReadoutDelta,
            metalFunctionName: "kv_readout_delta",
            bufferBindingCount: 6,
            constantCount: 1
        ),
        // 32: stateDecayUpdate — fused decay + outer product update
        //     state, g_val, key, delta + headDim
        SmeltKernelSignature(
            pipeline: .stateDecayUpdate,
            metalFunctionName: "state_decay_update",
            bufferBindingCount: 4,
            constantCount: 1
        ),
        // 33: affineMatvec — packed u4 weights, scales, biases, input, output + numRows
        //     cols and groupSize are now function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .affineMatvec,
            metalFunctionName: "affine_matvec",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 34: fusedLutMatvecAdd — indices, lut, input, output, residual + numRows
        //     cols and groupSize are now function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedLutMatvecAdd,
            metalFunctionName: "fused_lut_matvec_add",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 35: sigmoidMul — inputA, inputB, output + count
        SmeltKernelSignature(
            pipeline: .sigmoidMul,
            metalFunctionName: "sigmoid_mul",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 36: gatesKvReadoutDelta — fused gates + kv readout + delta
        //     state, key, value, b_proj, a_proj, a_log, dt_bias, delta, g_out + headDim
        SmeltKernelSignature(
            pipeline: .gatesKvReadoutDelta,
            metalFunctionName: "gates_kv_readout_delta",
            bufferBindingCount: 9,
            constantCount: 1
        ),
        // 37: fusedGateUpSwiglu — gate_indices, gate_lut, up_indices, up_lut, input, output + numRows
        //     cols and groupSize are function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedGateUpSwiglu,
            metalFunctionName: "fused_gate_up_swiglu",
            bufferBindingCount: 6,
            constantCount: 1
        ),
        // 38: fusedDualLutMatvec — w1_indices, w1_lut, w2_indices, w2_lut, input, output1, output2 + numRows
        //     cols and groupSize are function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedDualLutMatvec,
            metalFunctionName: "fused_dual_lut_matvec",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        // 39: fusedAffineMatvecAdd — weights, scales, biases, input, output, residual + numRows
        //     cols and groupSize are function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAdd,
            metalFunctionName: "fused_affine_matvec_add",
            bufferBindingCount: 6,
            constantCount: 1
        ),
        // 40: deltanetRecurrenceFusedDecode — mega-fused conv1d + L2norm + recurrence + gated_rms_norm
        //     state, qkv, b_proj, a_proj, A_log, dt_bias, output, convState, convWeight, z_proj, normWeight
        //     + headDim, headScale, qkvDim, convK, numHeads, rmsEps
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceFusedDecode,
            metalFunctionName: "deltanet_recurrence_fused",
            bufferBindingCount: 11,
            constantCount: 6
        ),
        // 41: fusedDualAffineMatvec — two affine matvecs sharing input
        //     w1_w, w1_s, w1_b, w2_w, w2_s, w2_b, input, out1, out2 + numRows
        //     cols and groupSize are function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvec,
            metalFunctionName: "fused_dual_affine_matvec",
            bufferBindingCount: 9,
            constantCount: 1
        ),
        // 42: fusedAffineGateUpSwiglu — two affine matvecs + SwiGLU
        //     gate_w, gate_s, gate_b, up_w, up_s, up_b, input, output + numRows
        //     cols and groupSize are function constants (FC_COLS, FC_GROUP_SIZE)
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwiglu,
            metalFunctionName: "fused_affine_gate_up_swiglu",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        // 43: fusedRmsNormAffineGateUpSwiglu — norm fused into FFN (sole consumer, no side-effect write)
        //     norm_input, norm_weight, gate_w, gate_s, gate_b, up_w, up_s, up_b, output + numRows, eps
        SmeltKernelSignature(
            pipeline: .fusedRmsNormAffineGateUpSwiglu,
            metalFunctionName: "fused_rms_norm_affine_gate_up_swiglu",
            bufferBindingCount: 9,
            constantCount: 2
        ),
        // 44: fusedRmsNormAffineMatvec — norm fused into first matvec (writes norm output as side effect)
        //     norm_input, norm_weight, norm_output, weights, scales, biases, output + numRows, eps
        SmeltKernelSignature(
            pipeline: .fusedRmsNormAffineMatvec,
            metalFunctionName: "fused_rms_norm_affine_matvec",
            bufferBindingCount: 7,
            constantCount: 2
        ),
        // 45: rmsNormScaleOnly — computes rsqrt scale only (1 TG, writes 4 bytes)
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnly,
            metalFunctionName: "rms_norm_scale_only",
            bufferBindingCount: 2,
            constantCount: 2
        ),
        // 46: normScaleAffineGateUpSwiglu — reads pre-computed scale, normalizes inline
        //     scale, norm_input, norm_weight, gate_w, gate_s, gate_b, up_w, up_s, up_b, output + numRows
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwiglu,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu",
            bufferBindingCount: 10,
            constantCount: 1
        ),
        // 47: normScaleAffineMatvec — reads pre-computed scale, normalizes inline, writes norm output
        //     scale, norm_input, norm_weight, norm_output, weights, scales, biases, output + numRows
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvec,
            metalFunctionName: "norm_scale_affine_matvec",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        // 48: atomicNormAffineGateUpSwiglu — fused norm+FFN via atomic signaling (sole consumer)
        //     norm_input, norm_weight, gate_w, gate_s, gate_b, up_w, up_s, up_b, output, scratch + numRows, eps
        SmeltKernelSignature(
            pipeline: .atomicNormAffineGateUpSwiglu,
            metalFunctionName: "atomic_norm_affine_gate_up_swiglu",
            bufferBindingCount: 10,
            constantCount: 2
        ),
        // 49: atomicNormAffineMatvec — fused norm+matvec via atomic signaling (writes norm output)
        //     norm_input, norm_weight, norm_output, weights, scales, biases, output, scratch + numRows, eps
        SmeltKernelSignature(
            pipeline: .atomicNormAffineMatvec,
            metalFunctionName: "atomic_norm_affine_matvec",
            bufferBindingCount: 8,
            constantCount: 2
        ),
        // 50: deltanetRecurrenceMlxDecode — split decode path recurrence core
        //     state[v,k], qkv, b_proj, a_proj, A_log, dt_bias, output + headDim, headScale, valueHeads, qkHeads
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxDecode,
            metalFunctionName: "deltanet_recurrence_mlx_decode",
            bufferBindingCount: 7,
            constantCount: 4
        ),
        // 51: conv1dUpdateSilu6144x4 — Qwen decode conv1d specialization
        //     conv_state[6144,4], qkv[6144], conv_weight[6144,4], qkv_out[6144]
        SmeltKernelSignature(
            pipeline: .conv1dUpdateSilu6144x4,
            metalFunctionName: "conv1d_update_silu_c6144_k4",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        // 52: l2NormalizeD128 — per-head L2 normalize specialization for headDim=128
        //     data[H,128] in-place
        SmeltKernelSignature(
            pipeline: .l2NormalizeD128,
            metalFunctionName: "l2_normalize_d128",
            bufferBindingCount: 1,
            constantCount: 0
        ),
        // 53: rmsNormGatedD128 — per-head gated RMS norm specialization for headDim=128
        //     input[H,128], gate[H,128], weight[128], output[H,128]
        SmeltKernelSignature(
            pipeline: .rmsNormGatedD128,
            metalFunctionName: "rms_norm_gated_d128",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        // 54: deltanetRecurrenceMlxDecodeD128H16 — Qwen tiled recurrence specialization
        //     state[16,128,128], qkv, b_proj, a_proj, A_log, dt_bias, output
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxDecodeD128H16,
            metalFunctionName: "deltanet_recurrence_mlx_decode_d128_h16",
            bufferBindingCount: 7,
            constantCount: 0
        ),
        // 55: attentionDecodeD256H8KV2 — Qwen decode attention specialization
        //     q_out[8,256] in-place, k_cache[2,256,256], v_cache[2,256,256] + seqLen
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV2,
            metalFunctionName: "attention_decode_d256_h8_kv2",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 56: rmsNorm1PWD2048 — fixed-shape decode RMSNorm specialization
        //     input[2048], weight[2048], output[2048]
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD2048,
            metalFunctionName: "rms_norm_1pw_d2048",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        // 57: fusedAffineGateUpSwigluC2048R6144G64 — Qwen FFN affine specialization
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2048R6144G64,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        // 58: fusedAffineMatvecAddC2048R2048G64 — Qwen output-proj residual fusion
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC2048R2048G64,
            metalFunctionName: "fused_affine_matvec_add_c2048_r2048_g64",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        // 59: fusedAffineMatvecAddC6144R2048G64 — Qwen FFN down-proj residual fusion
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC6144R2048G64,
            metalFunctionName: "fused_affine_matvec_add_c6144_r2048_g64",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        // 60: affineMatvecC2048R2048G64 — Qwen affine specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2048G64,
            metalFunctionName: "affine_matvec_c2048_r2048_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 61: affineMatvecC2048R6144G64 — Qwen QKV affine specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R6144G64,
            metalFunctionName: "affine_matvec_c2048_r6144_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 62: affineMatvecC2048R4096G64 — Qwen attention Q affine specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R4096G64,
            metalFunctionName: "affine_matvec_c2048_r4096_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 63: affineMatvecC2048R512G64 — Qwen attention KV affine specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R512G64,
            metalFunctionName: "affine_matvec_c2048_r512_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 64: affineEmbeddingGather — weights, scales, biases, tokenId, output + cols, groupSize
        SmeltKernelSignature(
            pipeline: .affineEmbeddingGather,
            metalFunctionName: "affine_embedding_gather",
            bufferBindingCount: 5,
            constantCount: 2
        ),
        // 65: fusedDualAffineMatvecC2048R16G64Batched — Qwen DeltaNet A+B prefill specialization
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2048R16G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2048_r16_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        // 66: fusedAffineGateUpSwigluC2048R6144G64Batched — Qwen FFN prefill specialization
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2048R6144G64Batched,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        // 67: affineMatvecC2048R2048G64Batched — Qwen 2048x2048 prefill specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2048G64Batched,
            metalFunctionName: "affine_matvec_c2048_r2048_g64_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 68: affineMatvecC2048R6144G64Batched — Qwen 2048x6144 prefill specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R6144G64Batched,
            metalFunctionName: "affine_matvec_c2048_r6144_g64_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 69: affineMatvecC2048R4096G64Batched — Qwen 2048x4096 prefill specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R4096G64Batched,
            metalFunctionName: "affine_matvec_c2048_r4096_g64_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 70: affineMatvecC2048R512G64Batched — Qwen 2048x512 prefill specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R512G64Batched,
            metalFunctionName: "affine_matvec_c2048_r512_g64_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 71: affineMatvecC6144R2048G64Batched — Qwen 6144x2048 prefill specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R2048G64Batched,
            metalFunctionName: "affine_matvec_c6144_r2048_g64_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 72: fusedDualAffineMatvecC2048R512G64Batched — Qwen attention K/V prefill specialization
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2048R512G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2048_r512_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        // 73: conv1dUpdateSilu6144x4Prefill — Qwen DeltaNet prompt conv over whole chunk
        SmeltKernelSignature(
            pipeline: .conv1dUpdateSilu6144x4Prefill,
            metalFunctionName: "conv1d_update_silu_c6144_k4_prefill",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        // 74: l2NormalizeQD128C6144H16Prefill — normalize Q slices in qkv[T,6144]
        SmeltKernelSignature(
            pipeline: .l2NormalizeQD128C6144H16Prefill,
            metalFunctionName: "l2_normalize_q_d128_c6144_h16_prefill",
            bufferBindingCount: 1,
            constantCount: 0
        ),
        // 75: l2NormalizeKD128C6144H16Prefill — normalize K slices in qkv[T,6144]
        SmeltKernelSignature(
            pipeline: .l2NormalizeKD128C6144H16Prefill,
            metalFunctionName: "l2_normalize_k_d128_c6144_h16_prefill",
            bufferBindingCount: 1,
            constantCount: 0
        ),
        // 76: deltanetRecurrenceMlxPrefillD128H16 — tiled prompt recurrence for Qwen
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxPrefillD128H16,
            metalFunctionName: "deltanet_recurrence_mlx_prefill_d128_h16",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        // 77: fusedAffineGateUpSwigluC2048R6144G64BatchedFull — qmm Qwen FFN prefill path
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2048R6144G64BatchedFull,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        // 78: affineMatvecC2048R2048G64BatchedFull — qmm Qwen 2048x2048 prefill path
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2048G64BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r2048_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 79: affineMatvecC2048R6144G64BatchedFull — qmm Qwen 2048x6144 prefill path
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R6144G64BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r6144_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 80: affineMatvecC2048R4096G64BatchedFull — qmm Qwen 2048x4096 prefill path
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R4096G64BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r4096_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 81: affineMatvecC2048R512G64BatchedFull — qmm Qwen 2048x512 prefill path
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R512G64BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r512_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 82: affineMatvecC6144R2048G64BatchedFull — qmm Qwen 6144x2048 prefill path
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R2048G64BatchedFull,
            metalFunctionName: "affine_matvec_c6144_r2048_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 83: affineEmbeddingGatherBatched — packed weights, scales, biases, tokenIds[B], output[B,hidden]
        //     + hidden, batchSize, groupSize
        SmeltKernelSignature(
            pipeline: .affineEmbeddingGatherBatched,
            metalFunctionName: "affine_embedding_gather_batched",
            bufferBindingCount: 5,
            constantCount: 3
        ),
        // 84: attentionDecodeD256H8KV2SDPA — MLX-style Qwen decode attention specialization
        //     q_out[8,256] in-place, k_cache[2,256,256], v_cache[2,256,256] + seqLen
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV2SDPA,
            metalFunctionName: "attention_decode_d256_h8_kv2_sdpa",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 85: perHeadRmsNorm1PWBatched — input[B,H,D], weight[D], output[B,H,D] + nHeads, headDim, eps
        SmeltKernelSignature(
            pipeline: .perHeadRmsNorm1PWBatched,
            metalFunctionName: "per_head_rms_norm_1pw_batched",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        // 86: affineMatvecC6144R2048G64 — Qwen FFN down-proj affine specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R2048G64,
            metalFunctionName: "affine_matvec_c6144_r2048_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 87: affineMatvecC2048R248320G64 — Qwen tied LM-head specialization
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R248320G64,
            metalFunctionName: "affine_matvec_c2048_r248320_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 88: fusedDualAffineMatvecC2048R16G64 — Qwen DeltaNet decode A+B specialization
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2048R16G64,
            metalFunctionName: "fused_dual_affine_matvec_c2048_r16_g64",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        // 89: conv1dUpdateSiluPrefill — generic DeltaNet prompt conv over whole chunk
        SmeltKernelSignature(
            pipeline: .conv1dUpdateSiluPrefill,
            metalFunctionName: "conv1d_update_silu_prefill",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        // 90: l2NormalizeQPrefill — generic prompt Q normalization over qkv[T,C]
        SmeltKernelSignature(
            pipeline: .l2NormalizeQPrefill,
            metalFunctionName: "l2_normalize_q_prefill",
            bufferBindingCount: 1,
            constantCount: 4
        ),
        // 91: l2NormalizeKPrefill — generic prompt K normalization over qkv[T,C]
        SmeltKernelSignature(
            pipeline: .l2NormalizeKPrefill,
            metalFunctionName: "l2_normalize_k_prefill",
            bufferBindingCount: 1,
            constantCount: 4
        ),
        // 92: deltanetRecurrenceMlxPrefill — generic tiled DeltaNet prompt recurrence
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxPrefill,
            metalFunctionName: "deltanet_recurrence_mlx_prefill",
            bufferBindingCount: 7,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1024,
            metalFunctionName: "rms_norm_1pw_d1024",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC1024R3584G64,
            metalFunctionName: "fused_affine_gate_up_swiglu_c1024_r3584_g64",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R2048G64,
            metalFunctionName: "affine_matvec_c1024_r2048_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R3584G64,
            metalFunctionName: "affine_matvec_c1024_r3584_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R4096G64,
            metalFunctionName: "affine_matvec_c1024_r4096_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R512G64,
            metalFunctionName: "affine_matvec_c1024_r512_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R6144G64,
            metalFunctionName: "affine_matvec_c1024_r6144_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1024G64,
            metalFunctionName: "affine_matvec_c2048_r1024_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R248320G64,
            metalFunctionName: "affine_matvec_c1024_r248320_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC1024R16G64,
            metalFunctionName: "fused_dual_affine_matvec_c1024_r16_g64",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC1024R512G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c1024_r512_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC1024R16G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c1024_r16_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC1024R3584G64BatchedFull,
            metalFunctionName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R2048G64BatchedFull,
            metalFunctionName: "affine_matvec_c1024_r2048_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R3584G64BatchedFull,
            metalFunctionName: "affine_matvec_c1024_r3584_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R4096G64BatchedFull,
            metalFunctionName: "affine_matvec_c1024_r4096_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R512G64BatchedFull,
            metalFunctionName: "affine_matvec_c1024_r512_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R6144G64BatchedFull,
            metalFunctionName: "affine_matvec_c1024_r6144_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1024G64BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r1024_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD2560,
            metalFunctionName: "rms_norm_1pw_d2560",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2560R9216G64,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2560_r9216_g64",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R8192G64,
            metalFunctionName: "affine_matvec_c2560_r8192_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R4096G64,
            metalFunctionName: "affine_matvec_c2560_r4096_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R1024G64,
            metalFunctionName: "affine_matvec_c2560_r1024_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R2560G64,
            metalFunctionName: "affine_matvec_c4096_r2560_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC9216R2560G64,
            metalFunctionName: "affine_matvec_c9216_r2560_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R248320G64,
            metalFunctionName: "affine_matvec_c2560_r248320_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2560R32G64,
            metalFunctionName: "fused_dual_affine_matvec_c2560_r32_g64",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2560R1024G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2560_r1024_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2560R32G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2560_r32_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2560R9216G64BatchedFull,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2560_r9216_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R8192G64BatchedFull,
            metalFunctionName: "affine_matvec_c2560_r8192_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R4096G64BatchedFull,
            metalFunctionName: "affine_matvec_c2560_r4096_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R1024G64BatchedFull,
            metalFunctionName: "affine_matvec_c2560_r1024_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R2560G64BatchedFull,
            metalFunctionName: "affine_matvec_c4096_r2560_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC9216R2560G64BatchedFull,
            metalFunctionName: "affine_matvec_c9216_r2560_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3584R1024G64,
            metalFunctionName: "affine_matvec_c3584_r1024_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3584R1024G64BatchedFull,
            metalFunctionName: "affine_matvec_c3584_r1024_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormGatedD128Batched,
            metalFunctionName: "rms_norm_gated_d128_batched",
            bufferBindingCount: 4,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1024Batched,
            metalFunctionName: "rms_norm_1pw_d1024_batched",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxDecodeD128H32QK16,
            metalFunctionName: "deltanet_recurrence_mlx_decode_d128_h32_qk16",
            bufferBindingCount: 7,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .deltanetRecurrenceMlxPrefillD128H32QK16,
            metalFunctionName: "deltanet_recurrence_mlx_prefill_d128_h32_qk16",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H16KV4,
            metalFunctionName: "attention_decode_d256_h16_kv4",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H16KV4SDPA,
            metalFunctionName: "attention_decode_d256_h16_kv4_sdpa",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R6144G64Rows4,
            metalFunctionName: "affine_matvec_c1024_r6144_g64_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1024G64Rows4,
            metalFunctionName: "affine_matvec_c2048_r1024_g64_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3584R1024G64Rows4,
            metalFunctionName: "affine_matvec_c3584_r1024_g64_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R2048G64Rows4,
            metalFunctionName: "affine_matvec_c1024_r2048_g64_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R248320G64Rows4,
            metalFunctionName: "affine_matvec_c1024_r248320_g64_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC1024R3584G64Rows4,
            metalFunctionName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .gegluFused,
            metalFunctionName: "geglu_fused",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .logitCap,
            metalFunctionName: "logit_cap",
            bufferBindingCount: 2,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .perHeadRmsNorm,
            metalFunctionName: "per_head_rms_norm",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .perHeadRmsNormNoScale,
            metalFunctionName: "per_head_rms_norm_noscale",
            bufferBindingCount: 1,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeSoftcap,
            metalFunctionName: "attention_decode_softcap",
            bufferBindingCount: 5,
            constantCount: 7
        ),
        SmeltKernelSignature(
            pipeline: .scalarMul,
            metalFunctionName: "scalar_mul",
            bufferBindingCount: 2,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .perHeadRmsNormBatched,
            metalFunctionName: "per_head_rms_norm_batched",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .perHeadRmsNormNoScaleBatched,
            metalFunctionName: "per_head_rms_norm_noscale_batched",
            bufferBindingCount: 1,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionPrefillSoftcap,
            metalFunctionName: "attention_prefill_softcap",
            bufferBindingCount: 4,
            constantCount: 8
        ),
        SmeltKernelSignature(
            pipeline: .fp16MatvecFP32Out,
            metalFunctionName: "fp16_matvec_fp32_out",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWFromFP32,
            metalFunctionName: "rms_norm_1pw_from_fp32",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .scalarMulWeight,
            metalFunctionName: "scalar_mul_weight",
            bufferBindingCount: 3,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWFromFP32Batched,
            metalFunctionName: "rms_norm_1pw_from_fp32_batched",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1536,
            metalFunctionName: "rms_norm_1pw_d1536",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1536Batched,
            metalFunctionName: "rms_norm_1pw_d1536_batched",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R2048G128,
            metalFunctionName: "affine_matvec_c1536_r2048_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R4096G128,
            metalFunctionName: "affine_matvec_c1536_r4096_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R256G128,
            metalFunctionName: "affine_matvec_c1536_r256_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R512G128,
            metalFunctionName: "affine_matvec_c1536_r512_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R6144G128,
            metalFunctionName: "affine_matvec_c1536_r6144_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R12288G128,
            metalFunctionName: "affine_matvec_c1536_r12288_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1536G128,
            metalFunctionName: "affine_matvec_c2048_r1536_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R1536G128,
            metalFunctionName: "affine_matvec_c4096_r1536_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R1536G128,
            metalFunctionName: "affine_matvec_c6144_r1536_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC12288R1536G128,
            metalFunctionName: "affine_matvec_c12288_r1536_g128",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC1536R256G128Batched,
            metalFunctionName: "fused_dual_affine_matvec_c1536_r256_g128_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC1536R512G128Batched,
            metalFunctionName: "fused_dual_affine_matvec_c1536_r512_g128_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R2048G128BatchedFull,
            metalFunctionName: "affine_matvec_c1536_r2048_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R4096G128BatchedFull,
            metalFunctionName: "affine_matvec_c1536_r4096_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R6144G128BatchedFull,
            metalFunctionName: "affine_matvec_c1536_r6144_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R12288G128BatchedFull,
            metalFunctionName: "affine_matvec_c1536_r12288_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1536G128BatchedFull,
            metalFunctionName: "affine_matvec_c2048_r1536_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R1536G128BatchedFull,
            metalFunctionName: "affine_matvec_c4096_r1536_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R1536G128BatchedFull,
            metalFunctionName: "affine_matvec_c6144_r1536_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC12288R1536G128BatchedFull,
            metalFunctionName: "affine_matvec_c12288_r1536_g128_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R2048G128Batched,
            metalFunctionName: "affine_matvec_c1536_r2048_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R4096G128Batched,
            metalFunctionName: "affine_matvec_c1536_r4096_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R6144G128Batched,
            metalFunctionName: "affine_matvec_c1536_r6144_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R12288G128Batched,
            metalFunctionName: "affine_matvec_c1536_r12288_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1536G128Batched,
            metalFunctionName: "affine_matvec_c2048_r1536_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R1536G128Batched,
            metalFunctionName: "affine_matvec_c4096_r1536_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R1536G128Batched,
            metalFunctionName: "affine_matvec_c6144_r1536_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC12288R1536G128Batched,
            metalFunctionName: "affine_matvec_c12288_r1536_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1,
            metalFunctionName: "attention_decode_d256_h8_kv1",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1SDPA,
            metalFunctionName: "attention_decode_d256_h8_kv1_sdpa",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1,
            metalFunctionName: "attention_decode_d512_h8_kv1",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1SDPA,
            metalFunctionName: "attention_decode_d512_h8_kv1_sdpa",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R2048G128Rows4,
            metalFunctionName: "affine_matvec_c1536_r2048_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R6144G128Rows4,
            metalFunctionName: "affine_matvec_c1536_r6144_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R12288G128Rows4,
            metalFunctionName: "affine_matvec_c1536_r12288_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R1536G128Rows4,
            metalFunctionName: "affine_matvec_c2048_r1536_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC6144R1536G128Rows4,
            metalFunctionName: "affine_matvec_c6144_r1536_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC12288R1536G128Rows4,
            metalFunctionName: "affine_matvec_c12288_r1536_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R262144G128Rows4,
            metalFunctionName: "affine_matvec_c1536_r262144_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1536R2048G128Rows4,
            metalFunctionName: "norm_scale_affine_matvec_c1536_r2048_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1536R12288G128Rows4,
            metalFunctionName: "norm_scale_affine_matvec_c1536_r12288_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1536R262144G128Rows4,
            metalFunctionName: "norm_scale_affine_matvec_c1536_r262144_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R256G128Rows4,
            metalFunctionName: "affine_matvec_c1536_r256_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC256R1536G128Rows4,
            metalFunctionName: "affine_matvec_c256_r1536_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1536Add,
            metalFunctionName: "rms_norm_1pw_d1536_add",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLU,
            metalFunctionName: "fused_affine_gate_up_geglu",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC1536R6144G128Rows4,
            metalFunctionName: "fused_affine_gate_up_geglu_c1536_r6144_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC1536R12288G128Rows4,
            metalFunctionName: "fused_affine_gate_up_geglu_c1536_r12288_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC2048R1536G128Rows4,
            metalFunctionName: "fused_affine_matvec_add_c2048_r1536_g128_rows4",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC4096R1536G128,
            metalFunctionName: "fused_affine_matvec_add_c4096_r1536_g128",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC6144R1536G128Rows4,
            metalFunctionName: "fused_affine_matvec_add_c6144_r1536_g128_rows4",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC12288R1536G128Rows4,
            metalFunctionName: "fused_affine_matvec_add_c12288_r1536_g128_rows4",
            bufferBindingCount: 6,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD1536,
            metalFunctionName: "rms_norm_scale_only_d1536",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC1536R6144G128Rows8,
            metalFunctionName: "fused_affine_gate_up_geglu_c1536_r6144_g128_rows8",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC1536R12288G128Rows8,
            metalFunctionName: "fused_affine_gate_up_geglu_c1536_r12288_g128_rows8",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R256G128Rows8,
            metalFunctionName: "affine_matvec_c1536_r256_g128_rows8",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpGeGLUC1536R6144G128Rows4,
            metalFunctionName: "norm_scale_affine_gate_up_geglu_c1536_r6144_g128_rows4",
            bufferBindingCount: 10,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpGeGLUC1536R12288G128Rows4,
            metalFunctionName: "norm_scale_affine_gate_up_geglu_c1536_r12288_g128_rows4",
            bufferBindingCount: 10,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1536R262144G128Rows8,
            metalFunctionName: "affine_matvec_c1536_r262144_g128_rows8",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1536R262144G128Rows8,
            metalFunctionName: "norm_scale_affine_matvec_c1536_r262144_g128_rows8",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1Fused,
            metalFunctionName: "attention_decode_d256_h8_kv1_fused",
            bufferBindingCount: 7,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1FusedSoftcap,
            metalFunctionName: "attention_decode_d256_h8_kv1_fused_softcap",
            bufferBindingCount: 7,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1FusedShared,
            metalFunctionName: "attention_decode_d256_h8_kv1_fused_shared",
            bufferBindingCount: 5,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV1FusedSharedSoftcap,
            metalFunctionName: "attention_decode_d256_h8_kv1_fused_shared_softcap",
            bufferBindingCount: 5,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1Fused,
            metalFunctionName: "attention_decode_d512_h8_kv1_fused",
            bufferBindingCount: 7,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1FusedSoftcap,
            metalFunctionName: "attention_decode_d512_h8_kv1_fused_softcap",
            bufferBindingCount: 7,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1FusedShared,
            metalFunctionName: "attention_decode_d512_h8_kv1_fused_shared",
            bufferBindingCount: 5,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV1FusedSharedSoftcap,
            metalFunctionName: "attention_decode_d512_h8_kv1_fused_shared_softcap",
            bufferBindingCount: 5,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC2048R8192G64BatchedFull,
            metalFunctionName: "fused_affine_gate_up_swiglu_c2048_r8192_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC8192R2048G64BatchedFull,
            metalFunctionName: "affine_matvec_c8192_r2048_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD1536AddBatched,
            metalFunctionName: "rms_norm_1pw_d1536_add_batched",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC2048R1024G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c2048_r1024_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC3584R1024G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c3584_r1024_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC2048R2048G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c2048_r2048_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC6144R2048G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c6144_r2048_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC8192R2048G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c8192_r2048_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC4096R2560G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c4096_r2560_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC9216R2560G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c9216_r2560_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD1024Batched,
            metalFunctionName: "rms_norm_scale_only_d1024_batched",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD2048Batched,
            metalFunctionName: "rms_norm_scale_only_d2048_batched",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD2560Batched,
            metalFunctionName: "rms_norm_scale_only_d2560_batched",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1024R6144G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c1024_r6144_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC1024R4096G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c1024_r4096_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2048R2048G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c2048_r2048_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2048R6144G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c2048_r6144_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2048R4096G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c2048_r4096_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2560R8192G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c2560_r8192_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwigluC1024R3584G64BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwigluC2048R6144G64BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwigluC2048R8192G64BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu_c2048_r8192_g64_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwigluC2560R9216G64BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu_c2560_r9216_g64_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD2048Eps1e5Batched,
            metalFunctionName: "rms_norm_scale_only_d2048_eps1e5_batched",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC3072R8192G64,
            metalFunctionName: "fused_affine_gate_up_swiglu_c3072_r8192_g64",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3072R3072G64,
            metalFunctionName: "affine_matvec_c3072_r3072_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3072R1024G64,
            metalFunctionName: "affine_matvec_c3072_r1024_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3072R8192G64,
            metalFunctionName: "affine_matvec_c3072_r8192_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC8192R3072G64,
            metalFunctionName: "affine_matvec_c8192_r3072_g64",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC3072R1024G64Batched,
            metalFunctionName: "fused_dual_affine_matvec_c3072_r1024_g64_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpSwigluC3072R8192G64BatchedFull,
            metalFunctionName: "fused_affine_gate_up_swiglu_c3072_r8192_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3072R3072G64BatchedFull,
            metalFunctionName: "affine_matvec_c3072_r3072_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC3072R1024G64BatchedFull,
            metalFunctionName: "affine_matvec_c3072_r1024_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC8192R3072G64BatchedFull,
            metalFunctionName: "affine_matvec_c8192_r3072_g64_batched_full",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC3072R3072G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c3072_r3072_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineMatvecAddC8192R3072G64BatchedFull,
            metalFunctionName: "fused_affine_matvec_add_c8192_r3072_g64_batched_full",
            bufferBindingCount: 7,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC3072R3072G64BatchedFull,
            metalFunctionName: "norm_scale_affine_matvec_c3072_r3072_g64_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpSwigluC3072R8192G64BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_swiglu_c3072_r8192_g64_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD3072Eps1e5Batched,
            metalFunctionName: "rms_norm_scale_only_d3072_eps1e5_batched",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD2560Add,
            metalFunctionName: "rms_norm_1pw_d2560_add",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD2560Batched,
            metalFunctionName: "rms_norm_1pw_d2560_batched",
            bufferBindingCount: 3,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R4096G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r4096_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R512G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r512_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R1024G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r1024_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10240G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r10240_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128Rows4,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R2560G128Rows4,
            metalFunctionName: "affine_matvec_c4096_r2560_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128Rows4,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R256G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r256_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC256R2560G128Rows4,
            metalFunctionName: "affine_matvec_c256_r2560_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R262144G128Rows8,
            metalFunctionName: "affine_matvec_c2560_r262144_g128_rows8",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC2560R10240G128Rows4,
            metalFunctionName: "fused_affine_gate_up_geglu_c2560_r10240_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2560R512G128Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2560_r512_g128_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedDualAffineMatvecC2560R1024G128Batched,
            metalFunctionName: "fused_dual_affine_matvec_c2560_r1024_g128_batched",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128Batched,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R4096G128Batched,
            metalFunctionName: "affine_matvec_c2560_r4096_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R512G128Batched,
            metalFunctionName: "affine_matvec_c2560_r512_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R1024G128Batched,
            metalFunctionName: "affine_matvec_c2560_r1024_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10240G128Batched,
            metalFunctionName: "affine_matvec_c2560_r10240_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128Batched,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R2560G128Batched,
            metalFunctionName: "affine_matvec_c4096_r2560_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128Batched,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R256G128Batched,
            metalFunctionName: "affine_matvec_c2560_r256_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC256R2560G128Batched,
            metalFunctionName: "affine_matvec_c256_r2560_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD2560AddBatched,
            metalFunctionName: "rms_norm_1pw_d2560_add_batched",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10752G128Rows4,
            metalFunctionName: "affine_matvec_c2560_r10752_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10752G128Batched,
            metalFunctionName: "affine_matvec_c2560_r10752_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNormScaleOnlyD2560,
            metalFunctionName: "rms_norm_scale_only_d2560",
            bufferBindingCount: 2,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpGeGLUC2560R10240G128Rows4,
            metalFunctionName: "norm_scale_affine_gate_up_geglu_c2560_r10240_g128_rows4",
            bufferBindingCount: 10,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normAddScaleAffineMatvecC2560R256G128Rows4,
            metalFunctionName: "norm_add_scale_affine_matvec_c2560_r256_g128_rows4",
            bufferBindingCount: 9,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC2560R10240G128Batched,
            metalFunctionName: "fused_affine_gate_up_geglu_c2560_r10240_g128_batched",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2560R10240G128Batched,
            metalFunctionName: "norm_scale_affine_matvec_c2560_r10240_g128_batched",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedNormRopeAndKvCachePrefill,
            metalFunctionName: "fused_norm_rope_and_kv_cache_prefill",
            bufferBindingCount: 7,
            constantCount: 9
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128BatchedTile4,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched_tile4",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC2560R10240G128BatchedFull,
            metalFunctionName: "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        // Reserved slots 302/303 — positional placeholders (see SmeltPipeline).
        // Never emitted by a shipped model; kept so later pipelines stay aligned.
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H8KV2SDPAReserved,
            metalFunctionName: "attention_decode_d256_h8_kv2_sdpa_reserved",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H8KV2SDPAReserved,
            metalFunctionName: "attention_decode_d512_h8_kv2_sdpa_reserved",
            bufferBindingCount: 3,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128Rows4SG1,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_rows4_sg1",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        // 305: clusterSparseLMHead — centroid_logits, lm_head_weight,
        //       hidden_state, token_ordering, logits_out
        //       + num_centroids, top_k, vocab_size, hidden_size,
        //         tokens_per_cluster, logit_cap
        // Metal source in Resources/Shaders/cluster.metal (unit 2c3).
        SmeltKernelSignature(
            pipeline: .clusterSparseLMHead,
            metalFunctionName: "cluster_sparse_lm_head",
            bufferBindingCount: 5,
            constantCount: 6
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC256R2048G128Rows4,
            metalFunctionName: "fused_affine_gate_up_geglu_c256_r2048_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpGeGLUC256R2048G128Rows4,
            metalFunctionName: "norm_scale_affine_gate_up_geglu_c256_r2048_g128_rows4",
            bufferBindingCount: 10,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC256R1024G128Rows4,
            metalFunctionName: "affine_matvec_c256_r1024_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC256R2048G128Rows4,
            metalFunctionName: "affine_matvec_c256_r2048_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC1024R256G128Rows4,
            metalFunctionName: "affine_matvec_c1024_r256_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R256G128Rows4,
            metalFunctionName: "affine_matvec_c2048_r256_g128_rows4",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD256H4KV2QnormRopeShared,
            metalFunctionName: "attention_decode_d256_h4_kv2_qnorm_rope_shared",
            bufferBindingCount: 6,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .attentionDecodeD512H4KV2QnormRopeShared,
            metalFunctionName: "attention_decode_d512_h4_kv2_qnorm_rope_shared",
            bufferBindingCount: 6,
            constantCount: 4
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineGateUpGeGLUC2560R10240G128BatchedFull,
            metalFunctionName: "norm_scale_affine_gate_up_geglu_c2560_r10240_g128_batched_full",
            bufferBindingCount: 11,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2560R2048G128Batched,
            metalFunctionName: "norm_scale_affine_matvec_c2560_r2048_g128_batched",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC2560R4096G128Batched,
            metalFunctionName: "norm_scale_affine_matvec_c2560_r4096_g128_batched",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .normScaleFusedDualAffineMatvecC2560R512G128Batched,
            metalFunctionName: "norm_scale_fused_dual_affine_matvec_c2560_r512_g128_batched",
            bufferBindingCount: 11,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleFusedDualAffineMatvecC2560R1024G128Batched,
            metalFunctionName: "norm_scale_fused_dual_affine_matvec_c2560_r1024_g128_batched",
            bufferBindingCount: 11,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecArgmaxC1536R262144G128Batched,
            metalFunctionName: "affine_matvec_argmax_c1536_r262144_g128_batched",
            bufferBindingCount: 5,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecArgmaxC2560R262144G128Batched,
            metalFunctionName: "affine_matvec_argmax_c2560_r262144_g128_batched",
            bufferBindingCount: 5,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .lmHeadArgmaxReduceR262144,
            metalFunctionName: "lm_head_argmax_reduce_r262144",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128BatchedSG4BT5,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_batched_sg4_bt5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .gegluFusedStridedBatched,
            metalFunctionName: "geglu_fused_strided_batched",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128BatchedSG4BT5,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_batched_sg4_bt5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128BatchedSG4BT5,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched_sg4_bt5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10240G128BatchedSG4BT5,
            metalFunctionName: "affine_matvec_c2560_r10240_g128_batched_sg4_bt5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD256Add,
            metalFunctionName: "rms_norm_1pw_d256_add",
            bufferBindingCount: 4,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .rmsNorm1PWD256AddScalarWeight,
            metalFunctionName: "rms_norm_1pw_d256_add_scalar_weight",
            bufferBindingCount: 5,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC256R1024G128Rows4,
            metalFunctionName: "norm_scale_affine_matvec_c256_r1024_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .normScaleAffineMatvecC256R2048G128Rows4,
            metalFunctionName: "norm_scale_affine_matvec_c256_r2048_g128_rows4",
            bufferBindingCount: 8,
            constantCount: 0
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10240G128BatchedExtB5,
            metalFunctionName: "affine_matvec_c2560_r10240_g128_batched_ext_b5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128BatchedExtB5,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched_ext_b5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .tqhEmbeddingGather,
            metalFunctionName: "tqh_embedding_gather",
            bufferBindingCount: 4,
            constantCount: 2
        ),
        SmeltKernelSignature(
            pipeline: .tqhMatvecPrepareInput,
            metalFunctionName: "tqh_matvec_prepare_input",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .tqhMatvec,
            metalFunctionName: "tqh_matvec",
            bufferBindingCount: 4,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .tqhMatvecPrepareInputBatched,
            metalFunctionName: "tqh_matvec_prepare_input_batched",
            bufferBindingCount: 2,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .tqhMatvecBatched,
            metalFunctionName: "tqh_matvec_batched",
            bufferBindingCount: 4,
            constantCount: 3
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128BatchedExtB5,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_batched_ext_b5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128BatchedExtB5,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_batched_ext_b5",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R10240G128BatchedExtB4,
            metalFunctionName: "affine_matvec_c2560_r10240_g128_batched_ext_b4",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128BatchedExtB4,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched_ext_b4",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128BatchedExtB4,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_batched_ext_b4",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128BatchedExtB4,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_batched_ext_b4",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .fusedAffineGateUpGeGLUC2560R10240G128BatchedBT4SG4,
            metalFunctionName: "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_bt4_sg4",
            bufferBindingCount: 8,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R262144G128Batched,
            metalFunctionName: "affine_matvec_c2560_r262144_g128_batched",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC10240R2560G128BatchedTile3,
            metalFunctionName: "affine_matvec_c10240_r2560_g128_batched_tile3",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R2048G128BatchedTile3,
            metalFunctionName: "affine_matvec_c2560_r2048_g128_batched_tile3",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2048R2560G128BatchedTile3,
            metalFunctionName: "affine_matvec_c2048_r2560_g128_batched_tile3",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC2560R4096G128BatchedTile3,
            metalFunctionName: "affine_matvec_c2560_r4096_g128_batched_tile3",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        SmeltKernelSignature(
            pipeline: .affineMatvecC4096R2560G128BatchedTile3,
            metalFunctionName: "affine_matvec_c4096_r2560_g128_batched_tile3",
            bufferBindingCount: 5,
            constantCount: 1
        ),
        // 351: snakeActivation — input, alpha, output + channels, length (Kokoro extraction)
        SmeltKernelSignature(
            pipeline: .snakeActivation,
            metalFunctionName: "snake_activation",
            bufferBindingCount: 3,
            constantCount: 2
        ),
        // 352: convTranspose1d — input, weight, bias, output + 6 conv params (Kokoro extraction)
        SmeltKernelSignature(
            pipeline: .convTranspose1d,
            metalFunctionName: "conv_transpose1d",
            bufferBindingCount: 4,
            constantCount: 6
        ),
        // 353: layerNorm — input, weight, bias, output + dim, eps (Kokoro extraction)
        SmeltKernelSignature(
            pipeline: .layerNorm,
            metalFunctionName: "layer_norm",
            bufferBindingCount: 4,
            constantCount: 2
        ),
        // 354: conv1dForward — input, weight, bias, output + 8 conv params (Kokoro
        // extraction; buf_stride dropped to fit the 8-constant dispatch cap)
        SmeltKernelSignature(
            pipeline: .conv1dForward,
            metalFunctionName: "conv1d_forward",
            bufferBindingCount: 4,
            constantCount: 8
        ),
        // 355-371: Qwen3-TTS fp32 kernels (buffer/constant counts match the kernel signatures).
        SmeltKernelSignature(pipeline: .snakeBetaF32, metalFunctionName: "snake_beta_f32", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .conv1dForwardF32, metalFunctionName: "conv1d_forward_f32", bufferBindingCount: 4, constantCount: 8),
        SmeltKernelSignature(pipeline: .convTranspose1dF32, metalFunctionName: "conv_transpose1d_f32", bufferBindingCount: 4, constantCount: 6),
        SmeltKernelSignature(pipeline: .layerNormCTF32, metalFunctionName: "layer_norm_ct_f32", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .matmulF32, metalFunctionName: "matmul_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .geluF32, metalFunctionName: "gelu_f32", bufferBindingCount: 2, constantCount: 1),
        SmeltKernelSignature(pipeline: .siluF32, metalFunctionName: "silu_f32", bufferBindingCount: 2, constantCount: 1),
        SmeltKernelSignature(pipeline: .swigluF32, metalFunctionName: "swiglu_f32", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .rmsNormCodecF32, metalFunctionName: "rms_norm_codec_f32", bufferBindingCount: 3, constantCount: 3),
        SmeltKernelSignature(pipeline: .rmsNormHeadF32, metalFunctionName: "rms_norm_head_f32", bufferBindingCount: 3, constantCount: 4),
        SmeltKernelSignature(pipeline: .ropeApplyF32, metalFunctionName: "rope_apply_f32", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .slidingAttnF32, metalFunctionName: "sliding_attn_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .causalGQAAttnF32, metalFunctionName: "causal_gqa_attn_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .rvqGatherSumF32, metalFunctionName: "rvq_gather_sum_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .scaleResidualF32, metalFunctionName: "scale_residual_f32", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .scaleResidualTCF32, metalFunctionName: "scale_residual_tc_f32", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .clampF32, metalFunctionName: "clamp_f32", bufferBindingCount: 2, constantCount: 3),
        SmeltKernelSignature(pipeline: .decodeGQAAttnF32, metalFunctionName: "decode_gqa_attn_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .matmulF16WF32, metalFunctionName: "matmul_f16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvF32, metalFunctionName: "gemv_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvF16WF32, metalFunctionName: "gemv_f16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .argmaxF32, metalFunctionName: "argmax_f32", bufferBindingCount: 2, constantCount: 2),
        SmeltKernelSignature(pipeline: .gatherRowF32, metalFunctionName: "gather_row_f32", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .gemmF32, metalFunctionName: "gemm_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemmF16WF32, metalFunctionName: "gemm_f16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .channelCopyF32, metalFunctionName: "channel_copy_f32", bufferBindingCount: 2, constantCount: 6),
        SmeltKernelSignature(pipeline: .nextFrameInputF32, metalFunctionName: "next_frame_input_f32", bufferBindingCount: 19, constantCount: 1),
        SmeltKernelSignature(pipeline: .cb0ArgmaxF32, metalFunctionName: "cb0_argmax_f32", bufferBindingCount: 3, constantCount: 7),
        SmeltKernelSignature(pipeline: .gemvBF16WF32, metalFunctionName: "gemv_bf16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemmBF16WF32, metalFunctionName: "gemm_bf16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .sampleTopKF32, metalFunctionName: "sample_topk_f32", bufferBindingCount: 3, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvU4F32, metalFunctionName: "gemv_u4_f32", bufferBindingCount: 6, constantCount: 5),
        SmeltKernelSignature(pipeline: .gemmU4F32, metalFunctionName: "gemm_u4_f32", bufferBindingCount: 6, constantCount: 5),
        SmeltKernelSignature(pipeline: .gatherRowBF16WF32, metalFunctionName: "gather_row_bf16w_f32", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .nextFrameInputBF16WF32, metalFunctionName: "next_frame_input_bf16w_f32", bufferBindingCount: 19, constantCount: 1),
        SmeltKernelSignature(pipeline: .gemmTNF32, metalFunctionName: "gemm_tn_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemmTNF16WF32, metalFunctionName: "gemm_tn_f16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemmTNBF16WF32, metalFunctionName: "gemm_tn_bf16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .causalGQAAttnSimdF32, metalFunctionName: "causal_gqa_attn_simd_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .slidingAttnSimdF32, metalFunctionName: "sliding_attn_simd_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvQKVBF16WF32, metalFunctionName: "gemv_qkv_bf16w_f32", bufferBindingCount: 7, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvGateUpSwigluBF16WF32, metalFunctionName: "gemv_gateup_swiglu_bf16w_f32", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .headNormRopeF32, metalFunctionName: "head_norm_rope_f32", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .gemvAddBF16WF32, metalFunctionName: "gemv_add_bf16w_f32", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .causalGQAAttnCachedF32, metalFunctionName: "causal_gqa_attn_cached_f32", bufferBindingCount: 4, constantCount: 5),
        SmeltKernelSignature(pipeline: .transposeF32, metalFunctionName: "transpose_f32", bufferBindingCount: 2, constantCount: 2),
        SmeltKernelSignature(pipeline: .cb0SampleTopKF32, metalFunctionName: "cb0_sample_topk_f32", bufferBindingCount: 4, constantCount: 9),
        SmeltKernelSignature(pipeline: .gatherRowsF32, metalFunctionName: "gather_rows_f32", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .gatherRowsBF16WF32, metalFunctionName: "gather_rows_bf16w_f32", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .causalGQAAttnCachedScalarF32, metalFunctionName: "causal_gqa_attn_cached_scalar_f32", bufferBindingCount: 4, constantCount: 5),
        // U2: fp16-activation dense matvec with bf16 / fp32 weights (weight, input, output + cols).
        SmeltKernelSignature(pipeline: .fp16MatvecBF16W, metalFunctionName: "fp16_matvec_bf16w", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .fp16MatvecFP32W, metalFunctionName: "fp16_matvec_fp32w", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .projectionBiasAddBatched, metalFunctionName: "projection_bias_add_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .attentionDecodeD128H16KV2SDPA, metalFunctionName: "attention_decode_d128_h16_kv2_sdpa", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .fusedAffineGateUpSwigluC2048R11008G64, metalFunctionName: "fused_affine_gate_up_swiglu_c2048_r11008_g64", bufferBindingCount: 8, constantCount: 0),
        SmeltKernelSignature(pipeline: .argmaxFP16Partials, metalFunctionName: "argmax_fp16_partials", bufferBindingCount: 2, constantCount: 2),
        SmeltKernelSignature(pipeline: .argmaxKeyReduce, metalFunctionName: "argmax_key_reduce", bufferBindingCount: 2, constantCount: 1),
        SmeltKernelSignature(pipeline: .affineMatvecC2048R151936G64Rows8, metalFunctionName: "affine_matvec_c2048_r151936_g64_rows8", bufferBindingCount: 5, constantCount: 0),
        SmeltKernelSignature(pipeline: .ropeKVCacheUpdate, metalFunctionName: "rope_kv_cache_update", bufferBindingCount: 4, constantCount: 8),
        SmeltKernelSignature(pipeline: .fusedAffineMatvecAddC2048R1024G64Rows4, metalFunctionName: "fused_affine_matvec_add_c2048_r1024_g64_rows4", bufferBindingCount: 6, constantCount: 0),
        SmeltKernelSignature(pipeline: .fusedDualAffineMatvecC1024R16G64Rows4, metalFunctionName: "fused_dual_affine_matvec_c1024_r16_g64_rows4", bufferBindingCount: 9, constantCount: 0),
        SmeltKernelSignature(pipeline: .conv1dUpdateSiluL2QKC6144K4D128H16, metalFunctionName: "conv1d_update_silu_l2_qk_c6144_k4_d128_h16", bufferBindingCount: 4, constantCount: 0),
        SmeltKernelSignature(pipeline: .conv1dUpdateSiluPrefillCheckpoint, metalFunctionName: "conv1d_update_silu_prefill_checkpoint", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .deltanetRecurrenceMlxPrefillCheckpoint, metalFunctionName: "deltanet_recurrence_mlx_prefill_checkpoint", bufferBindingCount: 8, constantCount: 4),
        SmeltKernelSignature(pipeline: .signedBinaryMatvecG128Rows8, metalFunctionName: "signed_binary_matvec_g128_rows8", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryMatvecG128Rows8, metalFunctionName: "signed_ternary_matvec_g128_rows8", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryEmbeddingGatherG128, metalFunctionName: "signed_binary_embedding_gather_g128", bufferBindingCount: 4, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedTernaryEmbeddingGatherG128, metalFunctionName: "signed_ternary_embedding_gather_g128", bufferBindingCount: 4, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryGateUpSwigluG128Rows8, metalFunctionName: "signed_binary_gate_up_swiglu_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryMatvecAddG128Rows8, metalFunctionName: "signed_binary_matvec_add_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .attentionDecodeD256H24KV4, metalFunctionName: "attention_decode_d256_h24_kv4", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryMatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_matvec_g128_rows8_batched_b4", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedTernaryMatvecG128Rows8BatchedB4, metalFunctionName: "signed_ternary_matvec_g128_rows8_batched_b4", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryGateUpSwigluG128Rows8BatchedB4, metalFunctionName: "signed_binary_gate_up_swiglu_g128_rows8_batched_b4", bufferBindingCount: 6, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryPackedBank4MatvecG128Rows8, metalFunctionName: "signed_binary_packed_bank4_matvec_g128_rows8", bufferBindingCount: 7, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI3G128, metalFunctionName: "signed_activation_bitplanes_i3_g128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3Bank4MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i3_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI2G128, metalFunctionName: "signed_activation_bitplanes_i2_g128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI2Bank4MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i2_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i3_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI2MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i2_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI4G128, metalFunctionName: "signed_activation_bitplanes_i4_g128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI4MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i4_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI4MatvecAddG128Rows8, metalFunctionName: "signed_binary_bitplane_i4_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3MatvecAddG128Rows8, metalFunctionName: "signed_binary_bitplane_i3_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI2MatvecAddG128Rows8, metalFunctionName: "signed_binary_bitplane_i2_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .normScaleSignedActivationBitplanesI4G128, metalFunctionName: "norm_scale_signed_activation_bitplanes_i4_g128", bufferBindingCount: 5, constantCount: 1),
        SmeltKernelSignature(pipeline: .normScaleSignedActivationBitplanesI3G128, metalFunctionName: "norm_scale_signed_activation_bitplanes_i3_g128", bufferBindingCount: 5, constantCount: 1),
        SmeltKernelSignature(pipeline: .normScaleSignedActivationBitplanesI2G128, metalFunctionName: "norm_scale_signed_activation_bitplanes_i2_g128", bufferBindingCount: 5, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI5G128, metalFunctionName: "signed_activation_bitplanes_i5_g128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI5MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i5_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI6G128, metalFunctionName: "signed_activation_bitplanes_i6_g128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI6MatvecG128Rows8, metalFunctionName: "signed_binary_bitplane_i6_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI5MatvecAddG128Rows8, metalFunctionName: "signed_binary_bitplane_i5_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI6MatvecAddG128Rows8, metalFunctionName: "signed_binary_bitplane_i6_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .rmsNormGatedD128SignedActivationBitplanesI6G128, metalFunctionName: "rms_norm_gated_d128_signed_activation_bitplanes_i6_g128", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .sigmoidMulSignedActivationBitplanesI6G128, metalFunctionName: "sigmoid_mul_signed_activation_bitplanes_i6_g128", bufferBindingCount: 4, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3GateUpSwigluG128Rows8, metalFunctionName: "signed_binary_bitplane_i3_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI4GateUpSwigluG128Rows8, metalFunctionName: "signed_binary_bitplane_i4_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI5GateUpSwigluG128Rows8, metalFunctionName: "signed_binary_bitplane_i5_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI6GateUpSwigluG128Rows8, metalFunctionName: "signed_binary_bitplane_i6_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .rmsScaleQK, metalFunctionName: "rms_scale_qk", bufferBindingCount: 1, constantCount: 4),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI4MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i4_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI5MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i5_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI6MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i6_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .swigluSignedActivationBitplanesI5G128, metalFunctionName: "swiglu_signed_activation_bitplanes_i5_g128", bufferBindingCount: 4, constantCount: 1),
        SmeltKernelSignature(pipeline: .normScaleSignedActivationBitplanesI5G128, metalFunctionName: "norm_scale_signed_activation_bitplanes_i5_g128", bufferBindingCount: 5, constantCount: 1),
        SmeltKernelSignature(pipeline: .normScaleSignedActivationBitplanesI6G128, metalFunctionName: "norm_scale_signed_activation_bitplanes_i6_g128", bufferBindingCount: 5, constantCount: 1),
        SmeltKernelSignature(pipeline: .rmsNormScaleOnlyPrecise, metalFunctionName: "rms_norm_scale_only_precise", bufferBindingCount: 2, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI4Bank4MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i4_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI5Bank4MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i5_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI6Bank4MatvecG128Rows8, metalFunctionName: "signed_ternary_bitplane_i6_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI4GateUpSwigluG128Rows8, metalFunctionName: "signed_ternary_bitplane_i4_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI5GateUpSwigluG128Rows8, metalFunctionName: "signed_ternary_bitplane_i5_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI6GateUpSwigluG128Rows8, metalFunctionName: "signed_ternary_bitplane_i6_gate_up_swiglu_g128_rows8", bufferBindingCount: 7, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI4MatvecG128Rows2Wide, metalFunctionName: "signed_ternary_bitplane_i4_matvec_g128_rows2_wide", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI5MatvecG128Rows2Wide, metalFunctionName: "signed_ternary_bitplane_i5_matvec_g128_rows2_wide", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryBitplaneI6MatvecG128Rows2Wide, metalFunctionName: "signed_ternary_bitplane_i6_matvec_g128_rows2_wide", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .residualAddRMSNormScaleOnlyPrecise, metalFunctionName: "residual_add_rms_norm_scale_only_precise", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .deltanetRecurrenceMlxDecodeD128H48QK16, metalFunctionName: "deltanet_recurrence_mlx_decode_d128_h48_qk16", bufferBindingCount: 7, constantCount: 0),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI2G128Batched, metalFunctionName: "signed_activation_bitplanes_i2_g128_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI3G128Batched, metalFunctionName: "signed_activation_bitplanes_i3_g128_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI4G128Batched, metalFunctionName: "signed_activation_bitplanes_i4_g128_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI5G128Batched, metalFunctionName: "signed_activation_bitplanes_i5_g128_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedActivationBitplanesI6G128Batched, metalFunctionName: "signed_activation_bitplanes_i6_g128_batched", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI2MatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_bitplane_i2_matvec_g128_rows8_batched_b4", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3MatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_bitplane_i3_matvec_g128_rows8_batched_b4", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI4MatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_bitplane_i4_matvec_g128_rows8_batched_b4", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI5MatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_bitplane_i5_matvec_g128_rows8_batched_b4", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI6MatvecG128Rows8BatchedB4, metalFunctionName: "signed_binary_bitplane_i6_matvec_g128_rows8_batched_b4", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI2MatvecG128Rows8BatchedB8, metalFunctionName: "signed_binary_bitplane_i2_matvec_g128_rows8_batched_b8", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI3MatvecG128Rows8BatchedB8, metalFunctionName: "signed_binary_bitplane_i3_matvec_g128_rows8_batched_b8", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI4MatvecG128Rows8BatchedB8, metalFunctionName: "signed_binary_bitplane_i4_matvec_g128_rows8_batched_b8", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI5MatvecG128Rows8BatchedB8, metalFunctionName: "signed_binary_bitplane_i5_matvec_g128_rows8_batched_b8", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .signedBinaryBitplaneI6MatvecG128Rows8BatchedB8, metalFunctionName: "signed_binary_bitplane_i6_matvec_g128_rows8_batched_b8", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .deltanetRecurrenceMlxPrefillD128H48QK16, metalFunctionName: "deltanet_recurrence_mlx_prefill_d128_h48_qk16", bufferBindingCount: 7, constantCount: 1),
        SmeltKernelSignature(pipeline: .attentionDecodeD256H24KV4SDPA, metalFunctionName: "attention_decode_d256_h24_kv4_sdpa", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryAffineMatvecG128Rows8, metalFunctionName: "signed_ternary_affine_matvec_g128_rows8", bufferBindingCount: 5, constantCount: 2),
        SmeltKernelSignature(pipeline: .ropeAndKvCachePrefillAnalytic, metalFunctionName: "rope_and_kv_cache_prefill_analytic", bufferBindingCount: 5, constantCount: 7),
        SmeltKernelSignature(pipeline: .attentionPrefillSDPAVectorD256, metalFunctionName: "attention_prefill_sdpa_vector_d256", bufferBindingCount: 4, constantCount: 7),
        SmeltKernelSignature(pipeline: .signedTernaryAffineQMMG128BM32BN32BK32, metalFunctionName: "signed_ternary_affine_qmm_g128_bm32_bn32_bk32", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .attentionPrefillMLXFallbackD256, metalFunctionName: "attention_prefill_mlx_fallback_d256", bufferBindingCount: 4, constantCount: 7),
        SmeltKernelSignature(pipeline: .attentionDecodeMLXVectorD256, metalFunctionName: "attention_decode_mlx_vector_d256", bufferBindingCount: 3, constantCount: 4),
        SmeltKernelSignature(pipeline: .attentionDecodeMLXVector2Pass1D256B128, metalFunctionName: "attention_decode_mlx_vector_2pass_1_d256_b128", bufferBindingCount: 5, constantCount: 4),
        SmeltKernelSignature(pipeline: .attentionDecodeMLXVector2Pass2D256B128, metalFunctionName: "attention_decode_mlx_vector_2pass_2_d256_b128", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .signedTernaryAffineGateUpSwigluG128Rows8, metalFunctionName: "signed_ternary_affine_gate_up_swiglu_g128_rows8", bufferBindingCount: 8, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryAffineMatvecAddG128Rows8, metalFunctionName: "signed_ternary_affine_matvec_add_g128_rows8", bufferBindingCount: 6, constantCount: 2),
        SmeltKernelSignature(pipeline: .signedTernaryAffineBank4MatvecG128Rows8, metalFunctionName: "signed_ternary_affine_bank4_matvec_g128_rows8", bufferBindingCount: 8, constantCount: 5),
        SmeltKernelSignature(pipeline: .noncausalAttentionF32, metalFunctionName: "noncausal_attention_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .layerNormRowsF32, metalFunctionName: "layer_norm_rows_f32", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .fourierPositionEmbeddingF32, metalFunctionName: "fourier_position_embedding_f32", bufferBindingCount: 2, constantCount: 7),
        SmeltKernelSignature(pipeline: .fsqBase8x5DecodeF32, metalFunctionName: "fsq_base8x5_decode_f32", bufferBindingCount: 2, constantCount: 1),
        SmeltKernelSignature(pipeline: .addRowsF32, metalFunctionName: "add_rows_f32", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .sigmoidF32, metalFunctionName: "sigmoid_f32", bufferBindingCount: 2, constantCount: 1),
        SmeltKernelSignature(pipeline: .denseBF16WF32, metalFunctionName: "dense_bf16w_f32", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .appendStridedFeaturesF32, metalFunctionName: "append_strided_features_f32", bufferBindingCount: 3, constantCount: 5),
        SmeltKernelSignature(pipeline: .layerNormRowsBF16WF32, metalFunctionName: "layer_norm_rows_bf16w_f32", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .extractInterleavedHeadPartF32, metalFunctionName: "extract_interleaved_head_part_f32", bufferBindingCount: 2, constantCount: 5),
        SmeltKernelSignature(pipeline: .rmsNormRowsBF16WF32, metalFunctionName: "rms_norm_rows_bf16w_f32", bufferBindingCount: 3, constantCount: 3),
        SmeltKernelSignature(pipeline: .repackConcatenatedHeadPartsF32, metalFunctionName: "repack_concatenated_head_parts_f32", bufferBindingCount: 6, constantCount: 4),
        SmeltKernelSignature(pipeline: .rmsNormCodecBF16WF32, metalFunctionName: "rms_norm_codec_bf16w_f32", bufferBindingCount: 3, constantCount: 3),
        SmeltKernelSignature(pipeline: .rmsNormHeadBF16WF32, metalFunctionName: "rms_norm_head_bf16w_f32", bufferBindingCount: 3, constantCount: 4),
        SmeltKernelSignature(pipeline: .headNormRopeBF16WF32, metalFunctionName: "head_norm_rope_bf16w_f32", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .pmpeBF16SemanticsF32, metalFunctionName: "pmpe_bf16_semantics_f32", bufferBindingCount: 2, constantCount: 7),
        SmeltKernelSignature(pipeline: .rmsNormCodecBF16, metalFunctionName: "rms_norm_codec_bf16", bufferBindingCount: 3, constantCount: 3),
        SmeltKernelSignature(pipeline: .rmsNormHeadBF16, metalFunctionName: "rms_norm_head_bf16", bufferBindingCount: 3, constantCount: 4),
        SmeltKernelSignature(pipeline: .headNormRopeBF16, metalFunctionName: "head_norm_rope_bf16", bufferBindingCount: 5, constantCount: 3),
        SmeltKernelSignature(pipeline: .gemvQKVBF16, metalFunctionName: "gemv_qkv_bf16", bufferBindingCount: 7, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemvAddBF16, metalFunctionName: "gemv_add_bf16", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .gemvGateUpSwigluBF16, metalFunctionName: "gemv_gateup_swiglu_bf16", bufferBindingCount: 4, constantCount: 2),
        SmeltKernelSignature(pipeline: .decodeGQAAttnBF16, metalFunctionName: "decode_gqa_attn_bf16", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .gemmBF16, metalFunctionName: "gemm_bf16", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .ropeApplyBF16, metalFunctionName: "rope_apply_bf16", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .causalGQAAttnCachedBF16, metalFunctionName: "causal_gqa_attn_cached_bf16", bufferBindingCount: 4, constantCount: 5),
        SmeltKernelSignature(pipeline: .scaleResidualTCBF16, metalFunctionName: "scale_residual_tc_bf16", bufferBindingCount: 4, constantCount: 3),
        SmeltKernelSignature(pipeline: .swigluBF16, metalFunctionName: "swiglu_bf16", bufferBindingCount: 3, constantCount: 1),
        SmeltKernelSignature(pipeline: .gatherRowBF16, metalFunctionName: "gather_row_bf16", bufferBindingCount: 3, constantCount: 2),
        SmeltKernelSignature(pipeline: .denseBF16, metalFunctionName: "dense_bf16", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .noncausalAttentionUpdateF32, metalFunctionName: "noncausal_attention_update_f32", bufferBindingCount: 6, constantCount: 7),
        SmeltKernelSignature(pipeline: .denseBF16WF32Rows4, metalFunctionName: "dense_bf16w_f32_rows4", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .denseBF16WF32Rows8, metalFunctionName: "dense_bf16w_f32_rows8", bufferBindingCount: 4, constantCount: 4),
        SmeltKernelSignature(pipeline: .denseBF16WF32Rows8Epilogue, metalFunctionName: "dense_bf16w_f32_rows8_epilogue", bufferBindingCount: 5, constantCount: 5),
        SmeltKernelSignature(pipeline: .noncausalAttentionQ8F32, metalFunctionName: "noncausal_attention_q8_f32", bufferBindingCount: 4, constantCount: 4),
    ]

    /// Look up a signature by pipeline case.
    public static func signature(for pipeline: SmeltPipeline) -> SmeltKernelSignature {
        signatures[pipeline.rawValue]
    }

    /// Metal function name to base catalog pipeline index.
    public static let pipelineIndexByName: [String: Int] = Dictionary(
        uniqueKeysWithValues: signatures.enumerated().map { index, signature in
            (signature.metalFunctionName, index)
        }
    )

    /// Look up a base catalog pipeline index by Metal function name.
    public static func pipelineIndex(named name: String) -> Int? {
        pipelineIndexByName[name]
    }

    /// Validate that the given buffer and constant counts match the catalog entry.
    /// Throws `SmeltCatalogError` if counts don't match.
    public static func validate(
        pipeline: SmeltPipeline,
        bufferCount: Int,
        constantCount: Int
    ) throws {
        let expected = signatures[pipeline.rawValue]
        let totalExpected = expected.bufferBindingCount + expected.constantCount
        let totalActual = bufferCount + constantCount
        if totalActual != totalExpected {
            throw SmeltCatalogError.bindingCountMismatch(
                pipeline: pipeline,
                expectedBuffers: expected.bufferBindingCount,
                expectedConstants: expected.constantCount,
                actualBuffers: bufferCount,
                actualConstants: constantCount
            )
        }
        if bufferCount != expected.bufferBindingCount {
            throw SmeltCatalogError.bufferCountMismatch(
                pipeline: pipeline,
                expected: expected.bufferBindingCount,
                actual: bufferCount
            )
        }
        if constantCount != expected.constantCount {
            throw SmeltCatalogError.constantCountMismatch(
                pipeline: pipeline,
                expected: expected.constantCount,
                actual: constantCount
            )
        }
    }

    /// Ordered list of Metal function names, suitable for the manifest's `pipelines` array.
    public static var pipelineNames: [String] {
        signatures.map(\.metalFunctionName)
    }
}

// MARK: - Errors

/// Errors thrown by catalog validation.
public enum SmeltCatalogError: Error, CustomStringConvertible {
    case bindingCountMismatch(
        pipeline: SmeltPipeline,
        expectedBuffers: Int,
        expectedConstants: Int,
        actualBuffers: Int,
        actualConstants: Int
    )
    case bufferCountMismatch(
        pipeline: SmeltPipeline,
        expected: Int,
        actual: Int
    )
    case constantCountMismatch(
        pipeline: SmeltPipeline,
        expected: Int,
        actual: Int
    )

    public var description: String {
        switch self {
        case let .bindingCountMismatch(
            pipeline, expectedBuf, expectedConst, actualBuf, actualConst
        ):
            return "Binding count mismatch for \(pipeline): "
                + "expected \(expectedBuf) buffers + \(expectedConst) constants, "
                + "got \(actualBuf) buffers + \(actualConst) constants"
        case let .bufferCountMismatch(pipeline, expected, actual):
            return "Buffer count mismatch for \(pipeline): "
                + "expected \(expected), got \(actual)"
        case let .constantCountMismatch(pipeline, expected, actual):
            return "Constant count mismatch for \(pipeline): "
                + "expected \(expected), got \(actual)"
        }
    }
}
