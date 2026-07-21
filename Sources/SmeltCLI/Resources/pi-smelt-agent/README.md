# Smelt agent for Pi

This directory supplies the terminal UI for two `smelt agent` flows:

- `index.ts` runs the `.agent` selected by `smelt agent run -i`.
- `author.ts` guides `smelt agent create <name>` and edits the source draft.

It is an implementation component shipped with Smelt, not a separate launcher
or model registry.

## Interactive run

```bash
smelt agent run triage -i
smelt agent run triage -i "start with this input"
```

The launcher resolves the agent, passes the absolute path of the current
`smelt` executable, assigns a stable local port, disables Pi discovery, and
replaces itself with Pi in the current terminal. The provider supervises
Smelt's HTTP transport for that package through the hidden
`smelt agent _serve-model` adapter. It rejects a server whose
`X-Smelt-Package-Identity` does not match the identity in `agent.json`.

Interactive requests identify the package-authored prompt contract as
`interactive/pi-v1`. A package may ship two independent prepared states:

```text
run/default        ordinary smelt agent run / API prompt prefix
interactive/pi-v1 Pi's fixed system and tool-schema prefix
```

The server restores a state only when both its contract ID and full token
prefix match. Requests without `prompt_contract` can use only `run/default`;
Pi can use only `interactive/pi-v1`. Request fields override the sampling
defaults stored with that contract.

The launcher passes these environment values:

```text
SMELT_AGENT_PI_AGENT_PACKAGE  absolute path to the selected .agent
SMELT_AGENT_PI_AGENT_ID       provider-local model id (normally current)
SMELT_AGENT_PI_AGENT_NAME     display name
SMELT_AGENT_PI_BIN            absolute path to the current smelt executable
SMELT_AGENT_PI_EXECUTABLE     pi executable override
SMELT_AGENT_PI_OPENAI_PORT    stable per-package local port
SMELT_AGENT_PI_SERVE_DIAGNOSTICS
                              defaults to 1; records cache/prefill timings
                              under ~/Library/Logs/smelt/agent/
```

`SMELT_AGENT_PI_EXTENSION_PATH` and `SMELT_AGENT_PI_EXECUTABLE` are
development/test overrides. `SMELT_AGENT_PI_BIN` is a required launcher
contract: a normal `smelt agent` invocation always sets it to the exact current
executable, so the extension never searches for another `smelt` on `PATH`.

## Conversational create

```bash
smelt agent create triage
```

The authoring extension is loaded alone, with Pi discovery and built-in tools
disabled. It can read or atomically write only these draft files:

```text
Agentfile
instructions.md
tools.json
cases.jsonl
```

`/try` and `/test` invoke the real `smelt agent create --from ...` and
`smelt agent run --once` paths. `/done` chooses the default run mode, creates
the final package, and exits only after success. Pi sessions live under the
draft's `.pi-sessions/` directory, so an interrupted authoring session resumes
without making its transcript the canonical source.

## Package check

```bash
npm pack --dry-run ./Sources/SmeltCLI/Resources/pi-smelt-agent
```

SwiftPM copies this directory into SmeltCLI's resource bundle. Packaged
distributions may alternatively install it under `share/smelt/agent/pi` beside
the `smelt` executable.
