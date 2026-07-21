# mmap_metal_cow

Proof that on Apple Silicon, `mmap` + `MTLDevice.makeBuffer(bytesNoCopy:)` gives us copy-on-write GPU memory and free disk persistence — the foundation for a paged-KV cache that uses the OS page cache as its inference cache.

## What was unknown

Conventional wisdom from CUDA-land says paged-KV (vLLM PagedAttention) requires explicit GPU block allocation and explicit copies for fork or persist. Apple Silicon's unified memory + Metal's `bytesNoCopy` API suggest a different shape:

1. mmap a file `MAP_PRIVATE`, wrap the pointer as an `MTLBuffer`, dispatch a Metal kernel that writes to it. **Does the GPU write trigger OS copy-on-write?** If yes, per-fork KV isolation is free.
2. mmap a file `MAP_SHARED`, GPU writes, msync. **Does the write reach disk?** If yes, KV persistence is `msync`.

Codex flagged the COW question as the architecture-defining unknown. This spike answers it.

## Results (M2 Max, macOS 15+)

```
page size:    16384 bytes
buffer size:  65536 bytes (4 pages)
device: Apple M2 Max

TEST A: GPU write to MAP_PRIVATE — does COW trigger?
  PASS post-write A[0] == 0xFF via CPU pointer
  PASS post-write B[0] == 0x42 — COW WORKED. GPU write triggered
       page fault, A diverged from B. Per-fork isolation via
       MAP_PRIVATE is real.
  PASS file on disk[0] == 0x42 — MAP_PRIVATE doesn't propagate to file
  PASS B[page 1, offset 0] == 0x42 — only the touched page diverged

TEST B: CPU write to MAP_PRIVATE — does standard COW work? (control)
  PASS baseline COW works

TEST C: GPU write to MAP_SHARED + msync — does it persist to disk?
  PASS file on disk[7] == 0xa7 via fresh fd. GPU writes survive
       to disk via MAP_SHARED + msync. Free KV persistence is real.
  PASS only written bytes changed

TEST D: timing
  mmap + makeBuffer(bytesNoCopy:): 4.4 µs avg
  device.makeBuffer(length:):      2.7 µs avg
  → 1.7 µs overhead per mmap'd buffer wrap. Negligible.
```

## What this unlocks

| Capability | How |
|------------|-----|
| Zero-cost session-tree forks | Each fork = a fresh `mmap(MAP_PRIVATE)` of the same KV file. GPU writes from any branch trigger per-page COW. Forks share physical pages until they diverge. |
| Free per-session persistence | `mmap(MAP_SHARED)` over a per-session KV file. Decode writes through the GPU. `msync` periodically; resume is `mmap` again. |
| Free prefix cache | Pre-warmed KV state for a system prompt or skill bundle lives as a regular file in `.smeltpkg/cache/`. Open session = `mmap(MAP_PRIVATE)`. Pages fault in lazily. Cold start is microseconds. |
| Free hot model swap | Already proven for weights. Same trick for KV blocks means model + KV can both be paged in/out via the OS page cache. |

## What this rules out

- We do **not** need `MTLHeap` aliasing. Codex was right that heap aliasing is not COW; we don't use it.
- We do **not** need to copy KV state across forks.
- We do **not** need to serialize KV state for persistence (it's already on disk).

## Architecture next

KV cache becomes a list of 16 KB page-aligned blocks per layer. Each block is its own `MTLBuffer` wrapping mmap'd memory:

- **Shared prefix blocks**: `mmap(MAP_PRIVATE)` over `prefix.kv` files. Read-mostly. Fork = open another mmap of the same file.
- **Per-session suffix blocks**: `mmap(MAP_SHARED)` over a per-session KV file. Append-only growth via GPU writes; periodic `msync`.

The attention kernel takes a block table (PagedAttention-style) that points across both. Fork is metadata-only. Persistence is free. Cold start pages in lazily. The OS does the refcounting.

## Run it yourself

```bash
cd spikes/mmap_metal_cow
swift run -c release MmapMetalCow
```

Requires macOS 15+, Apple Silicon, Swift 6.0+.

## File layout

```
spikes/mmap_metal_cow/
├── Package.swift
├── README.md
└── Sources/MmapMetalCow/main.swift
```

The Metal kernels are inline in `main.swift` (compiled at runtime via `device.makeLibrary(source:)`).
