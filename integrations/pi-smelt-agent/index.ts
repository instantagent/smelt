/**
 * Smelt agent provider extension for Pi.
 *
 * Pi calls into this extension whenever the user runs against a
 * packaged Smelt agent. We register a custom API type with pi-ai and delegate
 * actual chat-completion to pi-ai's built-in openai-completions
 * provider, talking to a locally-running Smelt transport over HTTP.
 *
 * Usage:
 *   SMELT_AGENT_PI_AGENT_PACKAGE=/path/to/triage.agent \
 *     pi -e ./integrations/pi-smelt-agent --model smelt-agent/current
 *
 * Environment:
 *   SMELT_AGENT_PI_OPENAI_HOST          (default 127.0.0.1) — Smelt transport bind host
 *   SMELT_AGENT_PI_OPENAI_PORT          (default 8080)      — Smelt transport port
 *   SMELT_AGENT_PI_OPENAI_AUTOSTART     (default true)      — auto-spawn transport if not reachable
 *   SMELT_AGENT_PI_SERVE_DIAGNOSTICS    (default true)      — log cache/prefill phases for spawned serve
 *   SMELT_AGENT_PI_AGENT_PACKAGE                            — package selected by smelt agent run
 *   SMELT_AGENT_PI_AGENT_ID             (default current)  — Pi model id for this package
 *   SMELT_AGENT_PI_AGENT_NAME                               — display name override
 *   SMELT_AGENT_PI_BIN                                      — current smelt binary path
 */
import { type ChildProcess, spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type ProviderResponse,
	type SimpleStreamOptions,
	streamSimpleOpenAICompletions,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

const PROVIDER = "smelt-agent";
const AGENT_API = "smelt-agent-native" as Api;
export const RUNTIME_CONFIGURABLE_CONTEXT_WINDOW = 2_147_483_647;

// --- Model registry --------------------------------------------------------

interface AgentModelConfig {
	id: string;
	name: string;
	packagePath: string;
	defaultMaxTokens: number;
}

interface RegisteredAgentModelConfig extends AgentModelConfig {
	contextWindow: number;
	maxTokens: number;
}

export const MODELS: AgentModelConfig[] = [{
	id: process.env.SMELT_AGENT_PI_AGENT_ID || "current",
	name: process.env.SMELT_AGENT_PI_AGENT_NAME || "Smelt agent",
	packagePath: process.env.SMELT_AGENT_PI_AGENT_PACKAGE || "",
	defaultMaxTokens: 512,
}];

const MODEL_MAP = new Map(MODELS.map((model) => [model.id, model]));

// --- Path resolution -------------------------------------------------------

function extensionDirectory(): string {
	return dirname(fileURLToPath(import.meta.url));
}

export function smeltHomeCandidates(): string[] {
	const extensionDir = extensionDirectory();
	return uniqueExistingOrConfiguredDirectories([
		process.env.SMELT_AGENT_PI_HOME,
		process.cwd(),
		resolve(process.cwd(), "..", "smelt"),
		resolve(extensionDir, "..", ".."),
		resolve(extensionDir, "..", "..", "..", "..", "..", "..", "smelt"),
	]);
}

function uniqueExistingOrConfiguredDirectories(paths: Array<string | undefined>): string[] {
	const seen = new Set<string>();
	const result: string[] = [];
	for (const path of paths) {
		if (!path) continue;
		const resolved = resolve(path);
		if (seen.has(resolved)) continue;
		seen.add(resolved);
		result.push(resolved);
	}
	return result;
}

export function resolveSmeltBin(): string {
	const configured = process.env.SMELT_AGENT_PI_BIN;
	if (configured) return configured;
	for (const home of smeltHomeCandidates()) {
		const candidate = join(home, ".build", "release", "smelt");
		if (existsSync(candidate)) return candidate;
	}
	return "smelt";
}

interface AgentManifest {
	version: number;
	name: string;
	model: { smelt_package_identity: string };
	instructions?: string | null;
	tools: string[];
	defaultMode: "once" | "interactive";
}

// Cache by package path: read once per process, keyed by absolute
// .agent path so identity is stable across model-id renames.
const manifestCache = new Map<string, AgentManifest>();

async function readAgentManifest(packagePath: string): Promise<AgentManifest> {
	const resolvedPackage = resolve(packagePath);
	const cached = manifestCache.get(resolvedPackage);
	if (cached) return cached;
	const raw = JSON.parse(
		await readFile(join(resolvedPackage, "agent.json"), "utf8"),
	) as unknown;
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new Error(`Invalid Smelt agent manifest at ${resolvedPackage}: expected object`);
	}
	const value = raw as Record<string, unknown>;
	if (value.version !== 1) {
		throw new Error(
			`Unsupported Smelt agent manifest version ${JSON.stringify(value.version)} at ${resolvedPackage}`,
		);
	}
	if (typeof value.name !== "string"
		|| !/^[\p{L}\p{N}][\p{L}\p{N}._-]*$/u.test(value.name)) {
		throw new Error(`Invalid Smelt agent name at ${resolvedPackage}`);
	}
	const model = value.model;
	if (typeof model !== "object" || model === null || Array.isArray(model)) {
		throw new Error(`Invalid Smelt agent model reference at ${resolvedPackage}`);
	}
	const identity = (model as Record<string, unknown>).smelt_package_identity;
	if (typeof identity !== "string" || !/^[0-9a-f]{64}$/.test(identity)) {
		throw new Error(`Invalid Smelt package identity at ${resolvedPackage}`);
	}
	if (value.instructions !== undefined && value.instructions !== null
		&& typeof value.instructions !== "string") {
		throw new Error(`Invalid Smelt agent instructions at ${resolvedPackage}`);
	}
	if (!Array.isArray(value.tools)
		|| !value.tools.every((tool) => typeof tool === "string"
			&& /^[\p{L}\p{N}][\p{L}\p{N}._-]*$/u.test(tool))
		|| new Set(value.tools).size !== value.tools.length) {
		throw new Error(`Invalid Smelt agent tools at ${resolvedPackage}`);
	}
	if (value.defaultMode !== "once" && value.defaultMode !== "interactive") {
		throw new Error(`Invalid Smelt agent defaultMode at ${resolvedPackage}`);
	}
	const manifest = value as unknown as AgentManifest;
	manifestCache.set(resolvedPackage, manifest);
	return manifest;
}

export async function resolveManifestModelIdentity(
	packagePath: string,
): Promise<string> {
	return (await readAgentManifest(packagePath)).model.smelt_package_identity;
}

export function resolvePackagePath(config: AgentModelConfig): string {
	if (!config.packagePath) {
		throw new Error(
			"SMELT_AGENT_PI_AGENT_PACKAGE is required; launch this extension through `smelt agent run -i`",
		);
	}
	return resolve(config.packagePath);
}

// --- Env helpers -----------------------------------------------------------

function envFlag(name: string, defaultValue: boolean): boolean {
	const value = process.env[name];
	if (value === undefined) return defaultValue;
	const normalized = value.trim().toLowerCase();
	if (["1", "true", "yes", "on"].includes(normalized)) return true;
	if (["0", "false", "no", "off"].includes(normalized)) return false;
	return defaultValue;
}

function envPositiveInteger(name: string): number | undefined {
	const raw = process.env[name];
	if (!raw) return undefined;
	const value = Number(raw);
	return Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function configuredContextLimit(): number | undefined {
	const configured = Number(process.env.SMELT_AGENT_PI_CONTEXT_LIMIT || "");
	return Number.isFinite(configured) && configured > 0 ? Math.floor(configured) : undefined;
}

function configuredMaxTokens(): number | undefined {
	return envPositiveInteger("SMELT_AGENT_PI_MAX_TOKENS");
}

function configuredTemperature(): number | undefined {
	const raw = process.env.SMELT_AGENT_PI_TEMPERATURE;
	if (raw === undefined || raw === "") return undefined;
	const value = Number(raw);
	if (!Number.isFinite(value) || value < 0) {
		throw new Error(`SMELT_AGENT_PI_TEMPERATURE must be non-negative, got ${JSON.stringify(raw)}`);
	}
	return value;
}

function configuredSeed(): number | undefined {
	const raw = process.env.SMELT_AGENT_PI_SEED;
	if (raw === undefined || raw === "") return undefined;
	const value = Number(raw);
	if (!Number.isSafeInteger(value) || value < 0) {
		throw new Error(`SMELT_AGENT_PI_SEED must be a non-negative safe integer, got ${JSON.stringify(raw)}`);
	}
	return value;
}

// --- HTTP-server auto-start ------------------------------------------------

interface AgentOpenAIServerHandle {
	bin: string;
	packagePath: string;
	port: number;
	host: string;
	child?: ChildProcess;
}

let smeltAgentOpenAIServer: AgentOpenAIServerHandle | undefined;
let smeltAgentOpenAIStartingPromise: Promise<void> | undefined;

function smeltAgentOpenAIPort(): number {
	return envPositiveInteger("SMELT_AGENT_PI_OPENAI_PORT") ?? 8080;
}

function smeltAgentOpenAIHost(): string {
	return process.env.SMELT_AGENT_PI_OPENAI_HOST || "127.0.0.1";
}

function smeltAgentOpenAIBaseUrl(): string {
	return `http://${smeltAgentOpenAIHost()}:${smeltAgentOpenAIPort()}/v1`;
}

function smeltAgentOpenAIAutostartEnabled(): boolean {
	return envFlag("SMELT_AGENT_PI_OPENAI_AUTOSTART", true);
}

/// Returns the path to the per-port Smelt transport auto-spawn log file,
/// creating its parent directory if needed. Defaults to
/// `~/Library/Logs/smelt/agent/serve-<port>.log`; overridable via
/// `SMELT_AGENT_PI_SERVE_LOG_DIR`.
function openServeLogFile(port: number): string {
	const dir = process.env.SMELT_AGENT_PI_SERVE_LOG_DIR
		|| join(homedir(), "Library", "Logs", "smelt", "agent");
	mkdirSync(dir, { recursive: true });
	return join(dir, `serve-${port}.log`);
}

async function isAgentOpenAIReachable(
	expectedPackageIdentity?: string,
	timeoutMs = 500,
): Promise<boolean> {
	try {
		const resp = await fetch(`${smeltAgentOpenAIBaseUrl()}/models`, {
			signal: AbortSignal.timeout(timeoutMs),
		});
		if (!resp.ok) return false;
		// Differentiate Smelt's /v1/models from a random local service and
		// bind the transport to the exact package referenced by the .agent.
		const body = (await resp.json()) as {
			data?: Array<{ id?: string; owned_by?: string }>;
		};
		const smeltEntries = (body?.data ?? []).filter(
			(entry) => entry?.owned_by === "smelt",
		);
		if (smeltEntries.length === 0) return false;
		const actualIdentity = resp.headers.get("x-smelt-package-identity");
		if (!actualIdentity || !/^[0-9a-f]{64}$/.test(actualIdentity)) return false;
		return expectedPackageIdentity === undefined
			|| actualIdentity === expectedPackageIdentity;
	} catch {
		return false;
	}
}

async function ensureAgentOpenAIServer(config: AgentModelConfig): Promise<void> {
	const requestedPackage = resolvePackagePath(config);

	// If we already auto-started a server, it serves exactly one
	// .agent. A request for a different model would 4xx, so fail
	// loud rather than silently misroute.
	if (smeltAgentOpenAIServer && smeltAgentOpenAIServer.packagePath !== requestedPackage) {
		throw new Error(
			"Smelt agent's Pi provider supports one resident model package per process "
			+ `(currently: ${smeltAgentOpenAIServer.packagePath}; `
			+ `requested: ${requestedPackage}). Set SMELT_AGENT_PI_OPENAI_AUTOSTART=0 `
			+ "and manage server processes manually for multi-model setups.",
		);
	}

	const expectedIdentity = await resolveManifestModelIdentity(requestedPackage);
	if (await isAgentOpenAIReachable(expectedIdentity)) return;
	if (await isAgentOpenAIReachable()) {
		throw new Error(
			`Smelt transport at ${smeltAgentOpenAIBaseUrl()} is running but does not `
			+ `serve the requested package (expected ${expectedIdentity} from `
			+ `${requestedPackage}). Stop the existing server `
			+ "(lsof -ti :<port> | xargs kill) or set SMELT_AGENT_PI_OPENAI_PORT to "
			+ "a different port.",
		);
	}
	if (!smeltAgentOpenAIAutostartEnabled()) {
		throw new Error(
			`Smelt transport unreachable at ${smeltAgentOpenAIBaseUrl()} and `
			+ "SMELT_AGENT_PI_OPENAI_AUTOSTART=0; start it manually with: "
			+ "smelt agent _serve-model <agent> --transport http --host 127.0.0.1 --port "
			+ String(smeltAgentOpenAIPort()),
		);
	}

	if (smeltAgentOpenAIStartingPromise) {
		await smeltAgentOpenAIStartingPromise;
		return;
	}

	smeltAgentOpenAIStartingPromise = (async () => {
		try {
			const bin = resolveSmeltBin();
			const port = smeltAgentOpenAIPort();
			const host = smeltAgentOpenAIHost();
			const args = [
				"agent", "_serve-model", requestedPackage,
				"--transport", "http",
				"--host", host,
				"--port", String(port),
			];
			const template = process.env.SMELT_AGENT_PI_TEMPLATE;
			if (template) args.push("--template", template);
			const contextLimit = configuredContextLimit();
			if (contextLimit !== undefined) {
				args.push("--context-limit", String(contextLimit));
			}
			// stderr: redirect to a per-port log file rather than
			// inheriting Pi's stderr. The detached transport must
			// not hold Pi's stderr fd open after Pi exits — if it
			// did, any parent process owning the read end of Pi's
			// stderr pipe (e.g., the homebrew gate scripts that
			// `result.stderr` capture Pi) would block forever waiting
			// for the pipe to EOF. Buffered "pipe" stdio would also
			// deadlock once the kernel buffer fills since nothing
			// consumes it, so an explicit log file is the durable
			// answer.
			// detached + unref(): keep the linked Smelt transport alive across Pi
			// process boundaries (model load is expensive; one-shot
			// `pi --print` invocations would otherwise re-load the
			// model every call) AND let Pi exit cleanly while the
			// child continues running. Users clean up the orphan with
			// `lsof -ti :<port> | xargs kill` or by killing the
			// process group.
			const logPath = openServeLogFile(port);
			const logFd = openSync(logPath, "a");
			const childEnv = { ...process.env };
			if (envFlag("SMELT_AGENT_PI_SERVE_DIAGNOSTICS", true)) {
				childEnv.SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS = "1";
			}
			const child = spawn(bin, args, {
				stdio: ["ignore", "ignore", logFd],
				detached: true,
				env: childEnv,
			});
			// Parent must close its dup so the transport's stderr
			// inherits only the child's reference.
			closeSync(logFd);
			child.unref();
			process.stderr.write(
				`smelt-agent-pi: started Smelt transport on port ${port}, `
				+ `logging to ${logPath}\n`,
			);
			child.on("error", (err) => {
				process.stderr.write(
					`smelt-agent-pi: failed to spawn Smelt transport: ${err.message}\n`,
				);
				smeltAgentOpenAIServer = undefined;
			});
			smeltAgentOpenAIServer = { bin, packagePath: requestedPackage, port, host, child };

			child.on("exit", (code) => {
				smeltAgentOpenAIServer = undefined;
				if (code !== null && code !== 0) {
					process.stderr.write(
						`smelt-agent-pi: Smelt transport exited with code ${code}\n`,
					);
				}
			});

			const deadline = Date.now() + 60_000;
			while (Date.now() < deadline) {
				if (await isAgentOpenAIReachable(expectedIdentity, 1_000)) return;
				await new Promise((r) => setTimeout(r, 250));
			}
			throw new Error(
				`Smelt transport did not become reachable at ${smeltAgentOpenAIBaseUrl()} within 60s`,
			);
		} finally {
			smeltAgentOpenAIStartingPromise = undefined;
		}
	})();

	await smeltAgentOpenAIStartingPromise;
}

// --- Pi session bridge -----------------------------------------------------

type AgentExtensionSessionManager = ExtensionContext["sessionManager"];
let extensionSessionManager: AgentExtensionSessionManager | undefined;

function bindExtensionSession(ctx: ExtensionContext): void {
	extensionSessionManager = ctx.sessionManager;
}

/// Stamp options.sessionId from Pi's session manager so the
/// session_id stays stable across the stream call even if Pi
/// switches sessions/branches mid-flight (the module-global
/// extensionSessionManager can mutate).
function enrichOptionsWithExtensionSession(
	_context: Context,
	options: SimpleStreamOptions | undefined,
): SimpleStreamOptions | undefined {
	if (!extensionSessionManager) return options;
	try {
		return {
			...options,
			sessionId: options?.sessionId ?? extensionSessionManager.getSessionId(),
		};
	} catch {
		return options;
	}
}

// --- Agent-OpenAI transport ------------------------------------------------

/// Map from Pi's session id to Agent's server-allocated session id
/// (returned in the X-Smelt-Session-Id header). The session id lets
/// the server identify a prefix-cache entry and surface 404 when
/// state is lost; Pi clears its persisted id on 404 and retries
/// with create_session: true.
const smeltAgentOpenAISessions = new Map<string, string>();

/// Rebrand the Pi-provided model as an openai-completions model
/// pointed at the locally-running Smelt transport. compat overrides
/// keep pi-ai from inferring support for store/developer-role/
/// reasoning_effort that the local server doesn't implement.
function buildAgentOpenAIModel(model: Model<Api>): Model<"openai-completions"> {
	return {
		...model,
		api: "openai-completions",
		baseUrl: smeltAgentOpenAIBaseUrl(),
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
		},
	};
}

interface AgentOpenAIPayload {
	session_id?: string;
	create_session?: boolean;
	prompt_contract?: string;
	[key: string]: unknown;
}

/// Set by onResponse when the server returns a 404. pi-ai's OpenAI
/// SDK currently throws on 4xx before onResponse fires, so this
/// flag stays false in practice; forwardOrSurfaceSession404 also
/// inspects the error event's message as the realistic signal.
interface SessionRetryState {
	saw404: boolean;
}

/// Returns false only if the FIRST event is a 404-tagged error;
/// caller retries with create_session: true. Once any non-error
/// event ships, the model has started streaming and a mid-stream
/// 404 wouldn't make sense, so errors propagate as-is.
async function forwardOrSurfaceSession404(
	upstream: AssistantMessageEventStream,
	target: AssistantMessageEventStream,
	output: AssistantMessage,
	retryState: SessionRetryState,
): Promise<boolean> {
	let sawNonError = false;
	for await (const event of upstream) {
		if (event.type === "start") {
			// streamAgent already emitted a start event before
			// dispatching here; forwarding pi-ai's would duplicate-
			// append the assistant turn.
			continue;
		}
		if (event.type === "error" && !sawNonError) {
			if (retryState.saw404
				|| (event.error.errorMessage?.includes("404") ?? false)) {
				return false;
			}
		}
		if (event.type !== "error") {
			sawNonError = true;
		}
		if (event.type === "done") {
			Object.assign(output, event.message);
		}
		target.push(event);
	}
	return true;
}

/// Resolve Pi's session id for this stream call. Per-request
/// options.sessionId wins over the module global because the global
/// can mutate mid-stream when Pi switches sessions/branches; the
/// per-request value was captured before the async work started.
function resolvePiSessionId(options: SimpleStreamOptions | undefined): string | null {
	return options?.sessionId ?? extensionSessionManager?.getSessionId() ?? null;
}

function smeltAgentOpenAIHooks(
	config: AgentModelConfig,
	piSessionId: string | null,
	forceCreate: boolean,
	retryState: SessionRetryState,
	upstream: Pick<SimpleStreamOptions, "onPayload" | "onResponse">,
): Pick<SimpleStreamOptions, "onPayload" | "onResponse"> {
	return {
		// Chain through upstream's onPayload first (callers may
		// transform the body themselves), THEN inject session_id /
		// create_session on the result. Same for onResponse — our
		// session-id capture runs alongside whatever the caller
		// also wants to observe.
		onPayload: async (payload, model) => {
			const upstreamResult = await upstream.onPayload?.(payload, model);
			const base = upstreamResult ?? payload;
			const persisted = !forceCreate && piSessionId
				? smeltAgentOpenAISessions.get(piSessionId)
				: undefined;
			const augmented: AgentOpenAIPayload = {
				...(base as Record<string, unknown>),
				prompt_contract: "interactive/pi-v1",
			};
			const manifest = await readAgentManifest(resolvePackagePath(config));
			const instructions = typeof manifest.instructions === "string"
				? manifest.instructions.trim() : "";
			if (instructions) {
				const messages = Array.isArray(augmented.messages)
					? [...augmented.messages] as Array<Record<string, unknown>> : [];
				const systemIndex = messages.findIndex((message) => message.role === "system");
				if (systemIndex < 0) {
					messages.unshift({ role: "system", content: instructions });
				} else {
					const existing = typeof messages[systemIndex].content === "string"
						? messages[systemIndex].content as string : "";
					messages[systemIndex] = {
						...messages[systemIndex],
						content: existing ? `${instructions}\n\n${existing}` : instructions,
					};
				}
				augmented.messages = messages;
			}
			const allowed = new Set(
				Array.isArray(manifest.tools)
					? manifest.tools.filter((tool): tool is string => typeof tool === "string")
					: [],
			);
			if (Array.isArray(augmented.tools)) {
				const requested = augmented.tools as Array<{
					function?: { name?: unknown };
				}>;
				const unauthorized = requested
					.map((tool) => tool.function?.name)
					.filter((name): name is string => typeof name === "string" && !allowed.has(name));
				if (unauthorized.length) {
					throw new Error(
						`Agent does not authorize tool(s): ${unauthorized.join(", ")}`,
					);
				}
			}
			const temperature = configuredTemperature();
			const seed = configuredSeed();
			if (temperature !== undefined) augmented.temperature = temperature;
			if (seed !== undefined) augmented.seed = seed;
			if (persisted) {
				augmented.session_id = persisted;
			} else {
				augmented.create_session = true;
			}
			if (process.env.SMELT_AGENT_PI_DEBUG_PAYLOAD === "1") {
				console.error("[smelt-agent-pi:onPayload]", JSON.stringify(augmented));
			}
			return augmented;
		},
		onResponse: async (response, model) => {
			await upstream.onResponse?.(response, model);
			if (response.status === 404 && !forceCreate && piSessionId) {
				smeltAgentOpenAISessions.delete(piSessionId);
				retryState.saw404 = true;
				return;
			}
			if (response.status < 400 && piSessionId) {
				const allocated = response.headers["x-smelt-session-id"];
				if (allocated) {
					smeltAgentOpenAISessions.set(piSessionId, allocated);
				}
			}
		},
	};
}

async function runAgentOpenAI(
	model: Model<Api>,
	config: AgentModelConfig,
	context: Context,
	options: SimpleStreamOptions | undefined,
	stream: AssistantMessageEventStream,
	output: AssistantMessage,
): Promise<void> {
	await ensureAgentOpenAIServer(config);
	const piSessionId = resolvePiSessionId(options);
	const openaiModel = buildAgentOpenAIModel(model);
	const firstState: SessionRetryState = { saw404: false };

	const firstHooks = smeltAgentOpenAIHooks(config, piSessionId, false, firstState, options ?? {});
	const firstAttempt = streamSimpleOpenAICompletions(openaiModel, context, {
		...options,
		...firstHooks,
	});
	const ok = await forwardOrSurfaceSession404(
		firstAttempt, stream, output, firstState,
	);
	if (ok) return;

	// First attempt errored with 404 before any tokens shipped
	// (stale session_id). Retry with create_session: true; the
	// smeltAgentOpenAISessions entry was already cleared in onResponse.
	const retryState: SessionRetryState = { saw404: false };
	const retryHooks = smeltAgentOpenAIHooks(config, piSessionId, true, retryState, options ?? {});
	const retryAttempt = streamSimpleOpenAICompletions(openaiModel, context, {
		...options,
		...retryHooks,
	});
	for await (const event of retryAttempt) {
		if (event.type === "start") continue;
		if (event.type === "done") {
			Object.assign(output, event.message);
		}
		stream.push(event);
	}
}

// --- Pi provider entry -----------------------------------------------------

function createEmptyAssistantMessage(model: Model<Api>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

async function resolveRegisteredModels(): Promise<RegisteredAgentModelConfig[]> {
	return Promise.all(
		MODELS.map(async (model) => {
			await readAgentManifest(resolvePackagePath(model));
			return {
				...model,
				contextWindow: configuredContextLimit() ?? RUNTIME_CONFIGURABLE_CONTEXT_WINDOW,
				maxTokens: configuredMaxTokens() ?? model.defaultMaxTokens,
			};
		}),
	);
}

export function streamAgent(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const output = createEmptyAssistantMessage(model);
	const effectiveOptions = enrichOptionsWithExtensionSession(context, options);

	(async () => {
		try {
			const config = MODEL_MAP.get(model.id);
			if (!config) throw new Error(`Unknown Smelt agent model: ${model.id}`);

			stream.push({ type: "start", partial: output });
			await runAgentOpenAI(model, config, context, effectiveOptions, stream, output);
		} catch (error) {
			output.stopReason = effectiveOptions?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({
				type: "error",
				reason: output.stopReason === "aborted" ? "aborted" : "error",
				error: output,
			});
		} finally {
			stream.end();
		}
	})();

	return stream;
}

export default async function (pi: ExtensionAPI) {
	const registeredModels = await resolveRegisteredModels();

	pi.on("session_start", async (_event, ctx) => {
		bindExtensionSession(ctx);
	});
	pi.on("session_tree", async (_event, ctx) => {
		bindExtensionSession(ctx);
	});
	pi.on("session_compact", async (_event, ctx) => {
		bindExtensionSession(ctx);
	});

	pi.registerCommand("smelt-agent-status", {
		description: "Show Smelt agent transport status",
		handler: async (_args, ctx) => {
			const packagePath = resolvePackagePath(MODELS[0]);
			const expectedIdentity = await resolveManifestModelIdentity(packagePath);
			const reachable = await isAgentOpenAIReachable(expectedIdentity, 1_000);
			const lines = [
				"Smelt agent transport: OpenAI completions to Smelt",
				`Smelt package ${expectedIdentity}: ${reachable ? "reachable" : "unreachable"} at ${smeltAgentOpenAIBaseUrl()}`,
				`Auto-start: ${smeltAgentOpenAIAutostartEnabled() ? "enabled" : "disabled"}`,
				`Active sessions: ${smeltAgentOpenAISessions.size}`,
				...(smeltAgentOpenAIServer ? [`Owned process: pid=${smeltAgentOpenAIServer.child?.pid ?? "?"} pkg=${smeltAgentOpenAIServer.packagePath}`] : []),
			];
			ctx.ui.setWidget("smelt-agent-status", lines, { placement: "belowEditor" });
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	pi.registerProvider(PROVIDER, {
		baseUrl: "file://smelt-agent-native",
		apiKey: "SMELT_AGENT_PI_API_KEY",
		api: AGENT_API,
		models: registeredModels.map((model) => ({
			id: model.id,
			name: model.name,
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: model.contextWindow,
			maxTokens: model.maxTokens,
		})),
		streamSimple: streamAgent,
	});
}
