# Contributing to Smelt

Smelt is an Apple Silicon model compiler/runtime toolkit. Changes should keep
model semantics package-authored, preserve artifact provenance, and measure
kernel/runtime work against the package actually executed.

Start with:

```console
swift build -c debug
bash tools/ci-lint.sh
```

The required local pre-merge gate for code changes is:

```console
bash tools/test-default.sh -c release --parallel --num-workers 3 --quiet
```

Run focused model, rig, package, performance, and maintainer gates when their
surface changes. GitHub CI deliberately stays a seconds-level Linux lint lane;
do not add hosted macOS, Metal, full-test, cache, or model jobs without owner
approval.

Instant Agent changes belong in `~/Projects/instantagent`. Smelt may expose a
generic API needed by that downstream consumer, but must never import its
product policy, registry, Pi integration, or `.agent` implementation.
