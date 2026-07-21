# spec_decode_acceptance

Empirical measurement of speculative-decode acceptance rate using the canonical Smelt artifacts. Answers: **is it worth implementing speculative decoding in Smelt?**

## Method

For each prompt P:

1. Run the **target** model (large) with `.argmax` selection. Capture the first 32 generated tokens as ground-truth sequence T = [t₀, t₁, ..., t₃₁].
2. For each position i in 0..32:
   - Form prefix = P + T[0..<i]
   - Run the **drafter** model (small) with `.argmax` on that prefix. Capture only the first generated token, d_i.
   - Accept = (d_i == t_i).
3. α (per-prompt acceptance) = matches / 32.

This is the standard spec-decode α: **probability that the drafter agrees with the target given the target's prior tokens.**

Also reported: **convergent prefix length** — how many initial tokens the drafter and target agree on without any teacher-forcing. This is the lower bound on speculative window before any rejection.

## Results — Llama 3.2 1B drafting 3B (M2 Max)

```
prompt                α       conv-len   notes
─────────────────────────────────────────────────────────────────────
factual               (skipped — target hit EOS at token 3: "Paris.")
code-completion       56.2%   4/32       partial code completion
json-tool-call        65.6%   4/32       JSON-shaped output
agent-style           75.0%   16/32      "Here are the steps to..." aligned
open-ended            62.5%   0/32       creative writing diverges immediately
code-explanation      65.6%   0/32       descriptive prose

overall α  = 65.0%
overall conv-len = 4.8 / 32 tokens
```

## Speedup model

Standard Leviathan et al. formula:

```
E[accepted] = (1 - α^(K+1)) / (1 - α)
```

At α = 0.65:

```
K=2 draft tokens → E[accepted] = 2.07 tokens/round
K=4 draft tokens → E[accepted] = 2.53 tokens/round
K=6 draft tokens → E[accepted] = 2.72 tokens/round
K=8 draft tokens → E[accepted] = 2.80 tokens/round
```

## Profitability

Effective speedup factors in draft cost:

```
speedup = E[accepted] / (1 + K · draft_cost / target_cost)
```

With Qwen baseline ratio 0.34 (3.2 ms / 9.5 ms):

```
K=2 → 1.24x  (win)
K=4 → 1.08x  (win, marginal)
K=6 → 0.90x  (loss)
K=8 → 0.76x  (loss)
```

## Verdict

**Speculative decoding is a marginal Tier 2 feature, not a Tier 1 killer.**

- Real-world gain at the optimal K: **~10-25% decode speedup**, not the 2-3x sometimes quoted.
- Codex predicted 0.4-0.65 for agentic chat; we measured 0.65. Codex was at the high end of the range and basically right.
- The win is bounded by cost ratio (1B isn't cheap enough vs 3B) and by per-position rejection compounding.
- Demote off the headline roadmap. Build it after KV-fork and constrained-tool-calls have shipped.

## Caveats

- The Qwen 3.5 family was unusable for this measurement on the current tree (decode emits whitespace only, likely a side effect of in-progress Gemma 4 fusion work in `Sources/SmeltCompiler/` and `Sources/SmeltRuntime/`). Once that lands and Qwen is rebuilt, re-run with Qwen 0.8B → 4B (5x cost ratio instead of 3x); expect slightly better profitability, not transformative.
- Per-position α overestimates true sequence-level acceptance. Real spec decode commits the target's chosen token after a rejection, putting the drafter back in unfamiliar territory; the per-token α tends to drop a few points after the first reset.
- `selectionMode: .argmax` only. Sampling at temperature > 0 changes the calculus (the drafter's distribution is more likely to overlap on top-k mass even when argmax disagrees).

## Build & run

```bash
cd spikes/spec_decode_acceptance
swift build -c release

# Sanity probe — both models in isolation
SPROBE=1 .build/release/SpecDecodeAcceptance

# Full measurement
.build/release/SpecDecodeAcceptance
```

Requires the canonical Llama 3.2 1B and 3B Smelt packages at:
- `artifacts/llama32-1b-affine/meta-llama_Llama-3.2-1B-Instruct.smeltpkg`
- `artifacts/llama32-3b-affine/meta-llama_Llama-3.2-3B-Instruct.smeltpkg`

Build them from the repo root with `bash tools/build-llama32-1b.sh && bash tools/build-llama32-3b.sh` if missing.
