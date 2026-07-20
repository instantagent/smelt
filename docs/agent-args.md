# Agent-declared CLI args (`args.json`)

A `.agent` can declare its own command-line interface. `ia run` exposes
the declared flags as first-class argv — typed, validated, documented — so an
installed agent behaves like a real program:

```console
$ ia run support-voice --help
...
Agent flags (declared by this package):
  --voice <Ryan|Katie>      Who speaks; default: Ryan
  --length <int>            Response budget in 80ms frames; default: 96

$ echo "Your order shipped today." | ia run support-voice --voice Katie
```

Design history and review trail:
`docs/plans/run-surface/run-args-unification-plan.md`. A working example for
the triage launch agent is `demo/triage-args.json`.

## The file

`args.json` at the package root. Targets are per package kind, so a TTS
interface and an LLM interface look different:

A TTS package — args alias the built-in voice fields:

```json
{
  "version": 1,
  "args": [
    {"flag": "voice", "type": "enum", "values": ["Ryan", "Katie"],
     "default": "Ryan", "target": "speaker", "description": "Who speaks"},
    {"flag": "length", "type": "int", "default": 96, "target": "max-frames",
     "description": "Response budget in 80ms frames"}
  ]
}
```

An LLM package — prompt-template placeholders + grammar bind slots:

```json
{
  "version": 1,
  "args": [
    {"flag": "service", "type": "string", "required": true,
     "description": "Service the report concerns"},
    {"flag": "routes", "type": "string-list", "target": "bind:routes",
     "description": "Allowed routing teams"}
  ],
  "prompt": "Service: {service}\n\nError report:\n{input}"
}
```

The same version-1 file can declare the agent's normal execution mode:

```json
{
  "version": 1,
  "run": { "defaultMode": "interactive" },
  "args": []
}
```

`defaultMode` is `interactive`, `once`, or `auto`. `ia run -i` and
`ia run --once` override it. Explicit prompt input selects one-shot execution;
`auto` opens Pi only when stdin and stdout are terminals. A run-only interface
is valid—`args` and `prompt` are optional.

Per arg: `flag` (exposed as `--flag`; `[a-z][a-z0-9-]*`), `type`
(`string | int | number | bool | enum | string-list`), `values` (enum only),
`default`, `required` (mutually exclusive with `default`), `description`,
`target`. A declared `bool` flag is presence-only on the CLI (`--verbose`
sets `true`; there is no `--verbose false` — that `false` would be prompt
text), so give bool args a `default` of `false`.

## Targets — where a flag's value lands

| Package kind | Target | Effect |
|---|---|---|
| TTS | `speaker`, `language`, `instruct`, `first-chunk`, `max-chunk`, `max-frames` | Aliases the built-in voice field (numeric targets require `int`) |
| LLM | *(omitted)* | Prompt-template placeholder: `{flag}` in `prompt` |
| LLM | `bind:NAME` | Feeds the baked grammar's `$bind:NAME` slot (requires `string-list`, CSV on the CLI) |

The `prompt` template (LLM only) substitutes `{flag}` for each declared
targetless arg and `{input}` for the positional/stdin text. Non-identifier
braces — JSON examples like `{"a": 1}` — pass through verbatim, but an
identifier-like placeholder that matches nothing (`{custmer}`) is a create
error: typos must not silently reach the model. Prompt-placeholder args must
be `required` or carry a `default` (the template must always be renderable).

## Resolution order

For TTS voice fields, one precedence chain:

```
explicit CLI flag  >  baked voice.json  >  args.json default  >  built-in default
```

LLM prompt-placeholder and bind args are simpler — explicit flag >
args.json default — because nothing bakes per-field values for them. The
baked prompt prefix stays its own persona path (prepended unless `--system`
overrides it), unchanged by declared args.

Equal-precedence conflicts are errors, never silent winners: `--voice` and
`--speaker` both given when one targets the other; a declared bind flag plus
an explicit `--bind` for the same slot.

## Strict argv

`ia run` rejects unknown `--flags` on both package kinds (a typo'd
`--speakr Ryan` used to be silently spoken). Escapes for literal flag-like
text: `--prompt "TEXT"`, or `--` — everything after it is prompt/speech text,
including for package resolution. Values are space-separated (`--flag=value`
gets a hint, not a guess). Only `--`-prefixed tokens are flags; `-5 degrees`
is just text.

## Creating an agent with an interface

```console
$ ia create voice --model base-voice.agent --args args.json
$ ia create triage --model base.agent --json-schema schema.json --args args.json
```

Create copies the base into a hidden sibling temporary, validates the complete
derived package, and atomically publishes the destination. It never mutates the
base or exposes a partially validated output. Validation covers shape and
collisions (declared names may not shadow built-ins), enum values and defaults
against the package's speaker/language tables (TTS), `bind:` targets against
the sealed grammar's slots (LLM), template placeholders, and the schedule a
flagless run would resolve across voice.json + args.json together. The run path
re-validates `args.json` on load, so a hand-edited file fails loudly instead of
changing flag semantics.

## Warm workers (`--linger N`) on both kinds

`--linger` now works on TTS packages with the same semantics as LLM: the
first call runs inline and leaves a detached worker holding the loaded model;
repeat calls forward over a Unix socket. Measured on the Qwen3-TTS 12hz
package: ~100 ms first audio steady-state warm vs ~2.5 s cold (load + TTFA).
Declared args resolve client-side on both kinds; only resolved primitives
cross the socket, so a warm worker can serve any flag combination. TTS
socket keys additionally rotate on voice.json/args.json changes; LLM keys
rotate on manifest/prefix/grammar changes and bind sets, as before.
