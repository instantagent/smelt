# Smelt vision

Smelt makes local model bring-up a coherent toolchain: author a model graph,
adapt checkpoints, compile a versioned package, execute it directly, keep it
resident behind a serving boundary, and inspect every correctness and
performance decision along the way.

The center of gravity is the compiled `.smeltpkg`. It is executable without an
agent framework and carries the package-authored graph, runtime policy,
modality adapters, kernels, weights, and integrity/provenance data required to
reproduce behavior.

Smelt should make three loops unusually tight:

1. Model bring-up: checkpoint to first trustworthy output.
2. Kernel work: first divergence to parity, then parity to measured speedup.
3. Product consumption: a stable Swift/runtime API without copied model code.

`smelt run` is the shortest one-request truth path. `smelt serve` is the
persistent form of the same package-faithful execution authority. Rigging,
text, audio, vision, and future modalities extend through declared block graphs
and typed ports rather than family switches in the public CLI.

The agent product is an optional leaf over the same lower-level toolkit.
`SmeltAgent` owns `.agent` overlays, Agentfile authoring, policy, registries,
tools, Pi UX, and behavioral evaluation; `smelt agent` exposes it through the
one installed executable. The dependency remains one-way inside the package:
the agent leaf consumes Smelt's runtime and content-addressed model packages,
while schema, compiler, runtime, serving, models, and lab code remain agent
independent. This keeps model and kernel gains universal without paying for a
second repository and release train.
