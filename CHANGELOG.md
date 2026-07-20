# Changelog

## 0.1.0 — 2026-07-14

The first public Instant Agent release.

- Create reusable `.agent` packages conversationally with `ia create`, or
  reproducibly from an `Agentfile`.
- Run the same package as a terminal conversation with `ia run -i` or as a
  deterministic Unix pipe stage with `ia run --once`.
- Declare each agent's default run mode, typed command-line arguments, tool
  allowlist, prompt template, and optional JSON-schema output contract.
- Restore prepared prefix state and compiled grammar vocabulary rather than
  rebuilding fixed work at request time.
- Install and publish thin, content-addressed agents with shared model blobs.
- Build and run the canonical Qwen 3.5 0.8B and 2B text packages and the
  Qwen3-TTS 0.6B CustomVoice package on Apple Silicon.
- Ship public cold-start, throughput, and structured-quality receipts.
