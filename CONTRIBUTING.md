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

Agent product changes belong in the isolated `SmeltAgent` target, the
`smelt agent` CLI leaf, and its Pi integration. Generic compiler, runtime,
serving, model, and lab targets must not import that layer. During the approved
integration, `~/Projects/instantagent` is a read-only source to migrate rather
than a second development authority.
