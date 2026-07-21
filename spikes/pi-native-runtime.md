# Pi-native Runtime — Implementation Plan

April 2026. Internal design doc.

## Thesis

**On Apple Silicon, the agent loop should run inside the model runtime, not the other way around.** Pi (the TypeScript coding agent) and Smelt (Swift+Metal inference) sharing one address space lets the KV cache, sampler, tool schemas, session tree, and prefix cache live in the same Metal heap. Cloud-API agents pay constant context-shipping overhead. A native pairing makes operations that are expensive in the cloud architecture (forks, persistence, prefix reuse, constrained sampling) effectively free.

This document is the engineering plan that follows from a day of derisks. Every load-bearing claim has a referenced spike. Numbers are real, measured on M2 Max.

---

## What today proved

| Layer | Question | Result | Spike |
|-------|----------|--------|-------|
| Memory | GPU write to `MAP_PRIVATE` mmap → OS copy-on-write? | Yes, per page | `mmap_metal_cow/` |
| Memory | `MAP_SHARED` + `msync` persists GPU writes to disk? | Yes | `mmap_metal_cow/` |
| Memory | Cold-page first-touch latency under GPU access | 2.66 µs / page → 4 MiB block warms in 0.7 ms | `mmap_metal_cow/` |
| Memory | RSS pressure under 100 simultaneous mmap forks | Bounded; 0.02 ms per fork open+wrap | `mmap_metal_cow/` |
| Compute | Block-table attention overhead vs flat-tensor | 1.04x avg (1.01 / 1.05 / 1.08 at N = 256 / 1024 / 4096) | `paged_attention/` |
| Compute | Mask compute cost under 5 ms decode budget | 0.2 µs on 257-vocab spike; ~50–70 µs predicted at 152k | `llguidance_wrap/` |
| Decode | Spec decode 1B → 3B Llama acceptance | α = 0.65 → ~1.24x at K=2 (marginal Tier 2) | `spec_decode_acceptance/` |
| Engine | llguidance C ABI works from Swift | Schema enforces enum branches at byte level | `llguidance_wrap/` |

What today **disproved**:

- **`MTLHeap` aliasing is not COW.** Apple's heap aliasing is memory reuse with undefined-on-overlap semantics. Use plain POSIX `mmap(MAP_PRIVATE)` + `bytesNoCopy` — the kernel does the COW for us in hardware-assisted page tables.
- **Speculative decoding is not a Tier 1 differentiator.** Real-world gain at the optimal K is 10–25%, not 2–3x. Demoted to Tier 2.
- **`smelt agent` (Pi inside `.smeltpkg`) is the inversion meme.** Niche distribution story, not a moat. Museum.
- **Building constrained-decode from scratch.** llguidance integration is days, not weeks; rolling our own buys nothing.

---

## The architecture

### Paged-mmap KV cache

KV cache is a list of fixed-size 16 KiB blocks per layer per (K|V). For Llama 3.2 3B (28 layers, 8 KV heads, head_dim 128, bf16): per-layer per-position K+V = 1024 bytes → **16 positions per 16 KiB block**. Page-aligned by construction.

Two block sources:

1. **Shared / persistent blocks** — `mmap(MAP_PRIVATE)` over a `.kv` file in a `.smeltpkg/cache/` or a per-session file. Read-mostly. Forks open additional `MAP_PRIVATE` mmaps of the same file. GPU writes trigger per-page COW.
2. **Per-fork suffix blocks** — `mmap(MAP_SHARED)` over a per-session file as the session grows. GPU appends KV directly. Periodic `msync` flushes to disk.

Each block becomes its own `MTLBuffer` via `device.makeBuffer(bytesNoCopy: ptr, length: 16384, options: .storageModeShared, deallocator: nil)`. The wrap costs ~1.7 µs per buffer (negligible). The attention kernel takes a block table — a small `[UInt32]` mapping logical block index → physical block index — and walks it with one indirection per BLOCK_SIZE inner positions. Overhead is 1.04x flat attention (measured).

Forking a session is `mmap` + a block-table copy: **~25 µs total**. Persisting a session is the OS's writeback machinery: **free, lazy, automatic**.

```
Session A           Session B (forked at turn 12)
  │                   │
  └─ mmap(MAP_PRIVATE)──> shared.kv ←──mmap(MAP_PRIVATE)─┘
       │                                  │
   block_table_A: [0,1,2,3,4,…]      block_table_B: [0,1,2,3,4,…]
       │                                  │
   GPU writes new turn 13             GPU writes new turn 13'
       │                                  │
   COW page for block 5               COW page for block 5
   (A sees its turn 13)               (B sees its turn 13')
```

### Constrained tool calls via llguidance

llguidance compiled as `aarch64-apple-darwin` static lib via `cargo build --release -p llguidance`. C header (`parser/llguidance.h`) wired to Swift via `module.modulemap` in a SwiftPM C target. Pre-existing engine, no fork, no patches.

Per decode step:

1. GPU produces logits for token `t`.
2. CPU thread runs `llg_matcher_compute_mask(matcher)` (~50 µs at our vocab) — *in parallel with* the GPU launch of the next step's matmul.
3. CPU AND's the 152k-bit mask into the 152k logits tensor (NEON `vbsl`, ~12 µs over 608 KB at 50 GB/s).
4. Sample top-p over the masked logits.
5. CPU calls `llg_matcher_consume_token(matcher, sampled_id)` to advance the parser.

Mask compute hides under the GPU step. Critical-path tail is ~150 µs of CPU work per token. At our 5 ms budget that's 3% — well below the threshold where it would extend latency.

Multi-tool dispatch is a top-level `anyOf` of registered tool schemas. llguidance's Earley parser collapses non-matching branches as soon as the `name` enum literal narrows. Verified in the spike: schema with `enum: ["read_file", "write_file"]` correctly masks down to `r|w` after the value-opening quote, then to `e` after `r`.

Pi `AssistantMessageEvent` stream is emitted by a Smelt-side `JSONStreamCursor` that watches the matcher's parser state and flushes a `toolcall_delta` at every JSON-structural boundary (closing `}`, `]`, `"`, value-end, property `,`).

### Persistent prefix cache

A `.smeltpkg/cache/` directory holds `.kv` files keyed by `hash(system_prompt + skills_bundle + project_context + model_id + tokenizer_id + sampler_id)`. Cold start of a Pi session:

1. Compute the cache key.
2. `mmap(MAP_PRIVATE)` the matching `.kv` file if it exists.
3. Wrap each block as an `MTLBuffer`.
4. Initialize the block table.
5. First decode step pages in only the blocks the GPU touches.

A cache hit costs ~25 µs of mmaps + first-touch faults at 2.66 µs/page on demand. A 100 MiB warmed prefix is fully accessible within a few ms of first use.

A cache miss runs prefill once, writes the resulting KV blocks to a new `.kv` file via `MAP_SHARED` + `msync`, registers it in the cache index. Future sessions hit warm. The cache is bounded by an LRU policy; eviction is `unlink` (the OS handles the rest).

**Invalidation matrix** (from codex's review):

| Change | Action |
|--------|--------|
| Model weights changed | Cache key includes `model_id` → automatic miss |
| Tokenizer changed | Cache key includes `tokenizer_id` → automatic miss |
| Sampler config changed | Cache key includes `sampler_id` → automatic miss |
| KV layout changed | Bumped `.smeltpkg` cache schema version → invalidates whole `cache/` |
| Prompt content changed | Hash mismatch → automatic miss |
| Skills set changed | Hash includes skill IDs → automatic miss |

v1 caches **system prompt + skills only** (the long, stable prefix). User-turn caching is v2.

### The N-API bridge to Pi

Pi (oh-my-pi) is a Bun monorepo with a Rust natives crate. Smelt links into it via:

```
┌──────────────────────────────────────────────┐
│  Bun host process                            │
│  └─ @oh-my-pi/pi-coding-agent SDK           │
│     └─ ModelRegistry registers `smelt:*`     │
│        └─ packages/ai provider stub          │
│           emits AssistantMessageEvent stream │
└────────────────┬─────────────────────────────┘
                 │  napi-rs threadsafe_function
┌────────────────▼─────────────────────────────┐
│  pi-smelt-natives (new Rust crate)           │
│  ─ links libSmeltBridge.a                    │
│  ─ exposes loadModel / generateStream(cb)    │
└────────────────┬─────────────────────────────┘
                 │  C ABI (@_cdecl Swift exports)
┌────────────────▼─────────────────────────────┐
│  SmeltBridge (new Swift static lib target)   │
│  ─ wraps SmeltModel + JSONStreamCursor       │
│  ─ converts callbacks to C function pointers │
└──────────────────────────────────────────────┘
        ↓
   SmeltRuntime / Metal kernels / paged KV
```

**Threading rules** (codex's flag, treat as load-bearing):

- One native worker queue owns Smelt sessions. JS talks through bounded queues.
- Every callback carries a generation id. Cancellation is idempotent.
- Finalizers only enqueue cleanup. Never tear down live Metal state synchronously.
- Swift code crossing Rust-created threads wraps in `autoreleasepool {}`.
- Backpressure: bounded queue depth between native worker and the TSFN. If TSFN backs up, the worker throttles, never drops.

This bridge is the smallest remaining derisk. A 200-line "hello world" Bun ↔ Swift round-trip with one `napi_threadsafe_function` callback per Metal command-buffer completion, before committing to the full provider integration.

---

## Phased roadmap

### Phase 0 — Tier 0: Provider parity (week 1)

The price of admission. Nothing exotic, just the bridge.

- [ ] **N-API bridge hello-world** (`spikes/napi_swift_bridge/`). Bun → napi-rs → Swift static lib → Metal kernel callback. Validates the threading model end-to-end.
- [ ] **`SmeltBridge` Swift static lib target** in main `Package.swift`. Wraps `SmeltModel`, exports `@_cdecl` C functions for `load`, `generate_stream`, `cancel`.
- [ ] **`pi-smelt-natives` Rust crate**. napi-rs bindings, exposes `SmeltSession` with async iterator over tokens.
- [ ] **`@oh-my-pi/smelt-provider` npm package**. Registers `api: smelt` in `packages/ai`, translates Smelt's per-token stream into `text_start → text_delta* → text_end → done` events.
- [ ] **`models.yml` registration**: `providers: { smelt: { api: smelt, models: [{ id: qwen35-2b, pkgPath: ... }] } }`.

**Gate:** A Pi session can chat with a `.smeltpkg` model end-to-end. Streaming works. Cancellation works. No tool calls yet.

### Phase 0.5 — Replay infrastructure (week 1–2, parallel)

The thing codex flagged as the most important missing piece. Build before any Tier 2 work.

- [ ] **Trace format**: per-decode-step JSONL with hashes for `prompt`, `model`, `tokenizer`, `sampler_config`, `grammar` (when applicable), `RNG_seed`, `accepted_token`, `mask_byte_count`, `kv_block_lineage`, `step_latency_us`.
- [ ] **`SmeltTracer`** Swift module gated by env var. Off by default; `SMELT_TRACE_DIR=...` enables.
- [ ] **`smelt replay <trace.jsonl>` CLI subcommand**. Re-runs a captured trace against a (possibly different) `.smeltpkg`, scores divergence per step. The release gate runs canonical Pi sessions through this.

**Gate:** Bug reports from real Pi sessions can be reproduced offline by replaying a trace.

### Phase 1 — Killer features (weeks 2–6)

Order matters. The KV work feeds the persistence work feeds the constrained-decode work.

#### 1a. Paged-mmap KV (weeks 2–3)

- [ ] **Block geometry** in `SmeltKernelShapeRegistry`. Family-specific block layout (Qwen / Llama / Gemma each get the right per-layer per-position math).
- [ ] **`SmeltKVBlockTable`** type. Owns block storage (mmap'd or anonymous), block-table buffer, and the runtime-side mapping.
- [ ] **Paged attention Metal kernel**. Generalize the spike's kernel: multi-head + GQA fan-out, real K/V dtype (bf16 or quantized), simdgroup-cooperative reduction.
- [ ] **Replace the existing flat-KV path** in `SmeltRuntime`. Old shape stays as a fallback gated by a manifest flag.
- [ ] **Verify gate parity**: same prompts → same tokens → same logits within fp16 tolerance. Run existing `verify-{qwen35,gemma4,llama32}-family.sh` scripts.

#### 1b. Free disk-backed prefix cache (week 4)

- [ ] **Cache key derivation** in `SmeltPrefixCache`: hash inputs as listed in the invalidation matrix.
- [ ] **Cache lookup + mmap-and-go** at session start.
- [ ] **Cache write**: after a cold prefill, blocks are already on `MAP_SHARED` pages; `msync` + register in `cache/index.json`.
- [ ] **LRU eviction** policy.
- [ ] **`smelt cache {list,clear,prune}`** CLI subcommands.

#### 1c. Constrained tool calls via llguidance (weeks 5–6)

- [ ] **Vendor llguidance** as part of main package build (SwiftPM plugin runs `cargo`, OR ship a checked-in `.xcframework` for release builds).
- [ ] **`SmeltTokenizerAdapter`** mapping `SmeltTokenizer` → `LlgTokenizerInitV2`. Critical: handle byte-fallback tokens correctly. Fuzz against the full Qwen 3.5 vocab.
- [ ] **`SmeltConstrainedSampler`** Swift wrapper. Compile-on-registration, mask-and-sample per step, NEON mask-apply.
- [ ] **`JSONStreamCursor`** state machine over llguidance parser state. Emits `toolcall_delta` at lexer-stable boundaries. Buffers up to a max-value-length (~1 KB).
- [ ] **Per-schema compile-time budget + relaxed-grammar fallback**. Critical for the predicted tail-latency failure mode.
- [ ] **Multi-tool grammar union**: `anyOf` of registered tool schemas built once at registration, reused.
- [ ] **Pi provider extension**: `toolcall_start` / `toolcall_delta` / `toolcall_end` events through the bridge.

**Gate:** Pi can call a real tool end-to-end with a guaranteed-valid JSON tool call. No retries. No parse errors.

### Phase 2 — Earn-it features (months 2–4, telemetry-driven)

Demote until telemetry from real Pi usage justifies them.

- **Speculative decoding** with Qwen 0.8B drafting 4B (when Qwen rebuild lands). Bench on real workloads first; α=0.65 model says ~1.24x at K=2.
- **KV-space compaction**. When Pi compacts a session, drop middle KV slots in place rather than re-prefill from compacted text.
- **Logprob / entropy event channel** as `AssistantMessageEvent` metadata. Skills opt in.
- **Real swarm via shared-weight parallel decode**. Multiple sub-agents share weights, ~250 MB extra KV per agent, no rate limit.
- **Hot model swap < 100 ms** via mmap reload.
- **Skills compiled into KV-fused `.smeltpkg` variants** (uses the existing `compile-time-fusion-aot-vision.md` direction).

### Phase 3 — Museum

- **`smelt agent`** (Pi inside `.smeltpkg`). Niche distribution story. Don't build.

---

## Predicted failures with mitigations

These are the things to expect bug reports about. Build the mitigation before shipping.

1. **Constrained-decode tail latency** — llguidance's 99.999th percentile is 10–30 ms on real anyOf+maxLength+enum schemas. At 200 tok/s that's a 6x budget blowout = visible stall. **Mitigation: per-schema compile-time budget + relaxed-grammar fallback path. Ship in v1.**

2. **Byte-fallback token misalignment.** Qwen splits multi-byte UTF-8 characters across tokens. The Smelt → llguidance tokenizer adapter is exactly where off-by-one bugs hide. **Mitigation: fuzz the full Qwen vocab at integration time. Add a vocab-roundtrip test in the verify suite.**

3. **Prefix-cache invalidation gaps.** The hash inputs list is comprehensive *as designed*. Real bugs come from a forgotten input — e.g., a Smelt compiler change that affects KV layout but doesn't bump the cache schema version. **Mitigation: cache schema version is a build-time constant in `SmeltSchema`, asserted on every cache load. Mismatch = silent miss + log line.**

4. **N-API + Bun lifecycle bugs.** Generation ID pattern handles most cases. The remaining risk is a Smelt session retained across a Bun event-loop shutdown that tries to fire a TSFN callback into a dead VM. **Mitigation: finalizer enqueues a "stop" message into the worker queue; worker drains all pending callbacks before exiting.**

5. **Many-fork RSS overhead in long-lived sessions.** The 100-fork stress test was bounded, but a 10,000-turn session with 50 active branches + COW pages may approach physical memory limits. **Mitigation: per-fork KV size telemetry; hard cap on simultaneously-live forks (configurable, default 32); LRU eviction of cold branches by `madvise(MADV_DONTNEED)` on their COW pages.**

---

## What's not in this plan

- Speculative decoding integration (Tier 2, telemetry-gated)
- KV-space compaction (Tier 2, telemetry-gated)
- Logprob event channel (Tier 2)
- Real swarm (Tier 2)
- `smelt agent` / Pi-inside-pkg (Tier 3, museum)
- xgrammar adoption (revisit only if llguidance tail latency is unfixable)
- Speculative-decode trace format (orthogonal; build if/when spec decode lands)
- Multi-modal (vision/audio) — separate plan

---

## Open derisks for later

These don't block Phase 0–1 but should be answered before Phase 2:

1. **Real-tokenizer mask cost at 152k vocab.** Spike used 257 vocab and saw 0.2 µs. Predicted scaling is 50–70 µs. Measure with the real `SmeltTokenizer` once Phase 1c lands.
2. **Cold-page latency under realistic page-fault parallelism.** Spike measured serial GPU-touch faults. With concurrent forks accessing different KV blocks, faults parallelize through the kernel — could be faster or slower. Bench with a real multi-session workload.
3. **Argument buffers for many `MTLBuffer`s.** If we go to per-block individual `MTLBuffer`s for very-large KV (hundreds of blocks), Metal's per-buffer binding limit becomes a constraint. Solution is `MTLArgumentEncoder`. Decide when block count regularly exceeds ~256.
4. **Constrained-decode + speculative decode interaction.** Spec drafter's predicted tokens must validate against the grammar mask. Drafter doesn't know the mask. Likely needs grammar-aware drafter or per-step mask intersection. Defer with spec decode itself.

---

## Operational hygiene note (Apr 2026)

Qwen 3.5 family decode is currently broken on the working tree (all sizes emit `[220, 220, ...]` for any prompt). Almost certainly a side effect of the in-progress Gemma 4 fusion changes in `Sources/SmeltCompiler/SmeltFusionPlanner.swift`, `Sources/SmeltRuntime/SmeltGemma4Profiles.swift`, and friends. Llama 3.2 still decodes fine. This needs to be fixed and Qwen packages rebuilt before any Phase 0 work that depends on Qwen — and certainly before the Gemma 4 work merges to `main`.

---

## Spike artifacts referenced

- `spikes/mmap_metal_cow/` — COW + persistence + page-fault + many-fork
- `spikes/paged_attention/` — flat vs paged Metal attention kernel
- `spikes/spec_decode_acceptance/` — α measurement, Llama 1B / 3B
- `spikes/llguidance_wrap/` — Swift integration of llguidance C ABI
- `spikes/constrained_decode_research.md` — engine landscape, build-vs-adopt rationale
- `spikes/compile-time-fusion-aot-vision.md` — feeds Phase 2 skill compilation
