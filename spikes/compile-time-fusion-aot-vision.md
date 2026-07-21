# Compile-Time Fusion And Autotuning Vision

## Summary

Smelt should move toward a compile-time region scheduler and autotuner that gives us tinygrad-like fusion leverage without adding a graph compiler to runtime.

The runtime goal stays simple:

- load a `.smeltpkg`
- execute the chosen dispatch graph
- avoid runtime graph rewriting, runtime search, and runtime kernel selection beyond cheap guarded branches

The compiler goal gets stronger:

- understand fusible regions, not just single ops
- generate multiple legal implementations for each region
- reject non-exact candidates against a staged quantized reference
- benchmark surviving candidates on the target machine
- record the winner in the built package

## Why This Exists

Manual shape packs got us good wins on Qwen and meaningful wins on Gemma, but they do not scale.

The current system is still mostly:

- local peephole fusion
- hand-picked shape specializations
- manual kernel selection by family and size

That is enough to prove the approach, but not enough to keep adding models without repeating the same work.

## Core Principle

For quantized kernels, `exact` means exact against the accepted quantized contract, not exact against FP16 or FP32.

Each optimized candidate must be compared to:

- the same packed weights
- the same scales and biases
- the same casts and rounding points
- the same staged affine or LUT computation

FP16 and FP32 remain model-quality references. They are not the oracle for whether a fused u4 kernel preserved semantics.

## Target Architecture

### 1. Richer Region IR

The compiler should lower decode and prefill into fusible regions, not only individual dispatches.

Initial region shapes:

- `rms_norm -> affine`
- `affine -> residual add`
- `gate/up -> activation -> down`
- attention score, softmax, and value accumulation subgraphs
- Gemma per-layer residual branch

### 2. Candidate Generation

For each region, the compiler should generate multiple equivalent implementations:

- staged reference
- partially fused variants
- fully fused variants
- shape-specialized variants
- tile and threadgroup variants
- variants that deliberately preserve half round-trips where required for exactness

### 3. Exactness Filter

Every candidate is tested against the staged quantized reference for the same region.

If a candidate changes outputs beyond the allowed exactness contract, it is discarded before benchmarking.

This is the hard boundary between:

- `implementation exactness`
- `model quality`

### 4. Compile-Time Benchmarking

Surviving candidates are microbenchmarked on the target Mac or on a device-family cache key.

The compiler records the fastest exact candidate for:

- op or region family
- model shape
- decode vs prefill regime
- GPU family

### 5. Package Plan Selection

The selected implementation is written into the `.smeltpkg` as:

- concrete pipeline list
- chosen dispatch graph
- optional tuning metadata and provenance

Runtime should not need to rediscover the plan.

## Relationship To tinygrad

The design target is similar in spirit to tinygrad, but different in when decisions happen.

tinygrad visibly relies on:

- graph scheduling
- kernel fusion
- codegen over equivalent kernels
- BEAM-like search for fast implementations

Smelt should copy that leverage at compile time instead of at runtime.

That is a better fit for Smelt because package build already knows:

- model topology
- tensor shapes
- quantization format
- decode vs prefill regimes
- shipping artifact boundaries

## Immediate Product Rules

Until the full system exists, Gemma and Qwen work should follow these rules:

1. Every hot fusion gets a local staged quantized oracle first.
2. End-to-end smoke is a confirmation step, not the first proof.
3. Performance tuning only happens after local exactness is established.
4. Trace and snapshot semantics are product constraints, not optional afterthoughts.

## Near-Term Build Order

### Phase 1: Exactness Harnesses

Add local staged quantized reference tests for the hottest decode chains:

- `affine_matvec + residual add`
- `rms_norm -> affine`
- Gemma GeGLU decode chain
- Gemma per-layer residual branch

### Phase 2: Region Selection

Replace more one-off peepholes with region-level lowering decisions in the compiler.

### Phase 3: Candidate Families

Teach the compiler to emit multiple legal implementations for the same region.

### Phase 4: AOT Tuning

Add a build-time tuner that benchmarks exact candidates and records the winner.

### Phase 5: Package Metadata

Persist tuning provenance so packages are self-describing and reproducible.

## Success Criteria

We should consider this direction successful when:

- new model families do not require large batches of hand-written one-off kernel wiring
- the compiler can choose between staged and fused implementations automatically
- exactness regressions are caught at the region level before runtime smoke
- package generation produces device-aware fast paths without runtime search

## Current Immediate Gemma Implication

Gemma should continue on the exact path first:

- restore the last known-good decode baseline
- prove hot decode chains against staged affine/u4 references
- then reduce dispatch count and retune from that exact base

The next tuning push should be driven by local staged quantized oracles, not by broad end-to-end fusion guesses.
