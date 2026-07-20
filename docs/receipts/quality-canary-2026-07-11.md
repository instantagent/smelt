# Qwen 3.5 structured-quality canary — 2026-07-11

This is the minimal quality receipt for the two text packages in the v0.1
release. It is a regression canary, not a leaderboard claim.

## Result

| Metric | Qwen 3.5 0.8B u4 | Qwen 3.5 2B u4 |
|---|---:|---:|
| Exact JSON object | 37/50 (74%) | 40/50 (80%) |
| Field accuracy | 146/165 (88.5%) | 153/165 (92.7%) |
| Valid JSON | 49/50 (98%) | 50/50 (100%) |

All 100 responses stopped on a package-declared terminator. No BF16 baseline
was run, so these results do not attribute errors to quantization.

## Method

- Hardware: Apple M2 Max, 32 GB unified memory; macOS 15.7.5.
- Binary: release `ia` built at private source commit `08faf23d`.
- Packages: canonical affine-u4, group-64 Qwen 3.5 0.8B and 2B artifacts.
- Corpus: 50 deterministic contact, order, event, biography, and tool-shaped
  extraction cases in
  [`tools/quality-canary/structured-corpus.jsonl`](../../tools/quality-canary/structured-corpus.jsonl).
- Runner: [`tools/quality-canary/run-structured.py`](../../tools/quality-canary/run-structured.py)
  against `ia serve`, temperature 0, maximum 256 output tokens.
- Scoring: the whole response must parse as one JSON object. String fields are
  compared after whitespace and case normalization; numeric fields must use a
  JSON number; an exact-object pass also requires the exact key set.

Reproduce after building a package:

```bash
tools/quality-canary/run-structured.sh path/to/model.agent
```

The checked-in corpus and runner make later scores directly comparable. The
full internal canary record also tracks GSM8K-lite, but that raw-completion
format is out of distribution for these chat-tuned packages and is not used as
a product claim.
