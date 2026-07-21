/** Conversational authoring extension launched by `smelt agent create <name>`. */
import { randomUUID } from "node:crypto";
import { appendFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";

const draft = requiredDirectory("SMELT_AGENT_AUTHOR_DRAFT");
const agentName = process.env.SMELT_AGENT_AUTHOR_NAME || "agent";
const outputPath = process.env.SMELT_AGENT_AUTHOR_OUTPUT || resolve(`${agentName}.agent`);
const smeltBin = requiredAbsolutePath("SMELT_AGENT_PI_BIN");

const draftFiles = [
	"Agentfile",
	"instructions.md",
	"tools.json",
	"cases.jsonl",
] as const;
type DraftFile = (typeof draftFiles)[number];

const DraftToolParams = Type.Object({
	action: StringEnum(["read", "write", "list_models"] as const),
	file: Type.Optional(StringEnum(draftFiles)),
	content: Type.Optional(Type.String()),
});

const CaseToolParams = Type.Object({
	action: StringEnum(["list", "add", "clear"] as const),
	input: Type.Optional(Type.String()),
	contains: Type.Optional(Type.String()),
	json: Type.Optional(Type.Boolean()),
});

interface Agentfile {
	version: number;
	model: string;
	system?: string;
	systemFile?: string;
	tools?: string[];
	defaultMode?: "interactive" | "once";
	[key: string]: unknown;
}

interface AgentCase {
	input: string;
	expect?: { contains?: string; json?: boolean };
}

function requiredDirectory(name: string): string {
	const value = process.env[name];
	if (!value) throw new Error(`${name} is required; launch through smelt agent create`);
	return resolve(value);
}

function requiredAbsolutePath(name: string): string {
	const value = process.env[name];
	if (!value) throw new Error(`${name} is required; launch through smelt agent create`);
	if (!isAbsolute(value)) throw new Error(`${name} must be the absolute path supplied by Smelt`);
	return value;
}

function pathFor(file: DraftFile): string {
	return join(draft, file);
}

async function readOptional(file: DraftFile): Promise<string> {
	try {
		return await readFile(pathFor(file), "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return "";
		throw error;
	}
}

async function atomicWrite(file: DraftFile, content: string): Promise<void> {
	await mkdir(draft, { recursive: true });
	const destination = pathFor(file);
	const temporary = join(draft, `.${file}.${randomUUID()}.tmp`);
	try {
		await writeFile(temporary, content, "utf8");
		await rename(temporary, destination);
	} catch (error) {
		await rm(temporary, { force: true });
		throw error;
	}
}

async function readAgentfile(): Promise<Agentfile> {
	return JSON.parse(await readFile(pathFor("Agentfile"), "utf8")) as Agentfile;
}

async function writeAgentfile(source: Agentfile): Promise<void> {
	validateAgentfile(source);
	await atomicWrite("Agentfile", `${JSON.stringify(source, null, 2)}\n`);
}

function validateAgentfile(source: Agentfile): void {
	if (source.version !== 1) throw new Error("Agentfile version must be 1");
	if (typeof source.model !== "string") throw new Error("Agentfile model must be a string");
	const allowed = new Set([
		"version", "model", "system", "systemFile", "tools", "defaultMode",
	]);
	const unknown = Object.keys(source).filter((key) => !allowed.has(key));
	if (unknown.length) throw new Error(`Unsupported Agentfile field(s): ${unknown.join(", ")}`);
	if (source.systemFile !== undefined
		&& (typeof source.systemFile !== "string" || source.systemFile.includes("/") || source.systemFile.includes(".."))) {
		throw new Error("systemFile must name a file inside the draft directory");
	}
	if (source.tools !== undefined
		&& (!Array.isArray(source.tools) || !source.tools.every((tool) => typeof tool === "string"))) {
		throw new Error("Agentfile tools must be an array of names");
	}
	if (source.defaultMode !== undefined && !["interactive", "once"].includes(source.defaultMode)) {
		throw new Error("Agentfile defaultMode must be interactive or once");
	}
}

function validateDraftFile(file: DraftFile, content: string): void {
	if (file === "Agentfile") {
		validateAgentfile(JSON.parse(content) as Agentfile);
		return;
	}
	if (file === "tools.json") {
		if (!content.trim()) return;
		const value = JSON.parse(content) as { version?: unknown; tools?: unknown };
		if (value.version !== 1 || !Array.isArray(value.tools) || value.tools.length === 0) {
			throw new Error("tools.json needs version 1 and a nonempty tools array");
		}
		const supported = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);
		if (!value.tools.every((tool) => typeof tool === "string" && supported.has(tool))) {
			throw new Error("tools.json contains an unsupported Pi tool");
		}
		if (new Set(value.tools).size !== value.tools.length) throw new Error("tools.json contains duplicates");
		return;
	}
	if (file === "cases.jsonl") {
		parseCases(content);
	}
}

async function writeDraftFile(file: DraftFile, content: string): Promise<void> {
	validateDraftFile(file, content);
	if (file === "Agentfile") {
		const source = JSON.parse(content) as Agentfile;
		source.version = 1;
		await writeAgentfile(source);
		return;
	}
	await atomicWrite(file, content);
	if (file === "instructions.md") {
		const source = await readAgentfile();
		if (content.trim()) source.systemFile = "instructions.md";
		else delete source.systemFile;
		await writeAgentfile(source);
	}
	if (file === "tools.json") {
		const source = await readAgentfile();
		if (content.trim()) {
			const value = JSON.parse(content) as { tools: string[] };
			source.tools = value.tools;
		} else {
			delete source.tools;
		}
		await writeAgentfile(source);
	}
}

async function installedModels(pi: ExtensionAPI): Promise<string[]> {
	const result = await pi.exec(smeltBin, ["agent", "_list-model-packages"], { timeout: 30_000 });
	if (result.code !== 0) {
		throw new Error(result.stderr.trim() || `smelt agent exited with code ${result.code}`);
	}
	return result.stdout.split("\n").map((line) => line.trim()).filter(Boolean);
}

function parseCases(content: string): AgentCase[] {
	return content.split("\n").filter((line) => line.trim()).map((line, index) => {
		const value = JSON.parse(line) as AgentCase;
		if (typeof value.input !== "string" || !value.input.trim()) {
			throw new Error(`cases.jsonl line ${index + 1} needs a nonempty input`);
		}
		return value;
	});
}

async function draftSummary(): Promise<string> {
	const source = await readAgentfile();
	const instructions = await readOptional("instructions.md");
	const tools = await readOptional("tools.json");
	const cases = parseCases(await readOptional("cases.jsonl"));
	return [
		`Agent: ${agentName}`,
		`Model: ${source.model || "not selected"}`,
		`Default: ${source.defaultMode || "once"}`,
		`Instructions: ${instructions.trim() ? `${instructions.trim().length} characters` : "not written"}`,
		`Interactive tools: ${tools.trim() ? "set" : "none"}`,
		`Saved cases: ${cases.length}`,
		`Draft: ${draft}`,
	].join("\n");
}

async function buildTryAgent(pi: ExtensionAPI): Promise<string> {
	const tryPath = join(draft, ".try.agent");
	const result = await pi.exec(
		smeltBin,
		["agent", "create", agentName, "--from", pathFor("Agentfile"), "--output", tryPath],
		{ cwd: draft, timeout: 10 * 60_000 },
	);
	if (result.code !== 0) {
		throw new Error((result.stderr || result.stdout || "smelt agent create failed").trim());
	}
	return tryPath;
}

async function setDefaultMode(mode: "interactive" | "once"): Promise<void> {
	const source = await readAgentfile();
	source.defaultMode = mode;
	await writeAgentfile(source);
}

function notifyResult(ctx: ExtensionContext, result: { stdout: string; stderr: string; code: number }): void {
	const message = [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n");
	ctx.ui.notify(message || `Exited ${result.code}`, result.code === 0 ? "info" : "error");
}

export default function authorAgent(pi: ExtensionAPI) {
	pi.registerTool({
		name: "agent_draft",
		label: "Agent draft",
		description: "Read or update one canonical agent source file, or list installed Smelt packages. Files are confined to this draft.",
		promptSnippet: "Read and update the current Smelt agent draft.",
		parameters: DraftToolParams,
		async execute(_id, params) {
			if (params.action === "list_models") {
				const models = await installedModels(pi);
				return { content: [{ type: "text", text: models.length ? models.join("\n") : "No installed Smelt packages found. Ask for a .smeltpkg path." }] };
			}
			if (!params.file) throw new Error("file is required for read and write");
			if (params.action === "read") {
				return { content: [{ type: "text", text: await readOptional(params.file) || "(empty)" }] };
			}
			if (params.content === undefined) throw new Error("content is required for write");
			await writeDraftFile(params.file, params.content);
			return { content: [{ type: "text", text: `Updated ${params.file}` }] };
		},
	});

	pi.registerTool({
		name: "agent_case",
		label: "Agent case",
		description: "List, add, or clear real examples used by /test.",
		promptSnippet: "Save representative examples for the agent test loop.",
		parameters: CaseToolParams,
		async execute(_id, params) {
			if (params.action === "list") {
				return { content: [{ type: "text", text: await readOptional("cases.jsonl") || "No saved cases" }] };
			}
			if (params.action === "clear") {
				await atomicWrite("cases.jsonl", "");
				return { content: [{ type: "text", text: "Cleared saved cases" }] };
			}
			if (!params.input?.trim()) throw new Error("input is required when adding a case");
			const value: AgentCase = { input: params.input };
			if (params.contains !== undefined || params.json !== undefined) {
				value.expect = { contains: params.contains, json: params.json };
			}
			await appendFile(pathFor("cases.jsonl"), `${JSON.stringify(value)}\n`, "utf8");
			return { content: [{ type: "text", text: "Saved case" }] };
		},
	});

	pi.registerCommand("spec", {
		description: "Show the current agent source summary",
		handler: async (_args, ctx) => {
			const summary = await draftSummary();
			ctx.ui.setWidget("agent-spec", summary.split("\n"), { placement: "belowEditor" });
			ctx.ui.notify(summary, "info");
		},
	});

	pi.registerCommand("try", {
		description: "Build the draft and run one real example",
		handler: async (input, ctx) => {
			const prompt = input.trim() || await ctx.ui.input("Try the agent", "Example input");
			if (!prompt) return;
			try {
				ctx.ui.setStatus("agent-create", "building draft…");
				const packagePath = await buildTryAgent(pi);
				ctx.ui.setStatus("agent-create", "running draft…");
				const result = await pi.exec(smeltBin, ["agent", "run", packagePath, "--once", prompt], {
					cwd: draft,
					timeout: 10 * 60_000,
				});
				notifyResult(ctx, result);
			} catch (error) {
				ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
			} finally {
				ctx.ui.setStatus("agent-create", undefined);
			}
		},
	});

	pi.registerCommand("test", {
		description: "Build the draft and run every saved case",
		handler: async (_args, ctx) => {
			try {
				const cases = parseCases(await readOptional("cases.jsonl"));
				if (!cases.length) throw new Error("No cases saved. Add examples while we talk, then run /test.");
				ctx.ui.setStatus("agent-create", `testing ${cases.length} case(s)…`);
				const packagePath = await buildTryAgent(pi);
				const lines: string[] = [];
				let failures = 0;
				for (const [index, test] of cases.entries()) {
					const result = await pi.exec(smeltBin, ["agent", "run", packagePath, "--once", test.input], {
						cwd: draft,
						timeout: 10 * 60_000,
					});
					let failure = result.code === 0 ? "" : `exit ${result.code}`;
					if (!failure && test.expect?.contains && !result.stdout.includes(test.expect.contains)) {
						failure = `missing ${JSON.stringify(test.expect.contains)}`;
					}
					if (!failure && test.expect?.json) {
						try { JSON.parse(result.stdout); } catch { failure = "output is not JSON"; }
					}
					if (failure) failures += 1;
					lines.push(`${failure ? "✗" : "✓"} ${index + 1}: ${failure || "passed"}`);
				}
				ctx.ui.notify(`${lines.join("\n")}\n\n${cases.length - failures}/${cases.length} passed`, failures ? "error" : "info");
			} catch (error) {
				ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
			} finally {
				ctx.ui.setStatus("agent-create", undefined);
			}
		},
	});

	pi.registerCommand("done", {
		description: "Review the run mode and create the final .agent",
		handler: async (_args, ctx) => {
			const choice = await ctx.ui.select(
				"Should this normally be a conversation or a command?",
				["Conversation", "Command"],
			);
			if (!choice) return;
			await setDefaultMode(choice === "Conversation" ? "interactive" : "once");
			const summary = await draftSummary();
			const confirmed = await ctx.ui.confirm("Create agent", `${summary}\n\nOutput: ${outputPath}`);
			if (!confirmed) return;
			ctx.ui.setStatus("agent-create", "creating final agent…");
			const result = await pi.exec(
				smeltBin,
				["agent", "create", agentName, "--from", pathFor("Agentfile"), "--output", outputPath],
				{ cwd: draft, timeout: 10 * 60_000 },
			);
			ctx.ui.setStatus("agent-create", undefined);
			notifyResult(ctx, result);
			if (result.code === 0) ctx.shutdown();
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		pi.setSessionName(`create ${agentName}`);
		ctx.ui.setTitle(`Create ${agentName} — Smelt agent`);
		ctx.ui.setWidget("agent-create", [
			`Creating ${agentName}`,
			"/spec  /try  /test  /done",
			`Source: ${draft}`,
		], { placement: "belowEditor" });
	});

	pi.on("before_agent_start", async (event) => {
		const summary = await draftSummary();
		return {
			systemPrompt: `${event.systemPrompt}\n\n${AUTHORING_PROMPT}\n\nCurrent draft:\n${summary}`,
		};
	});
}

const AUTHORING_PROMPT = `You help someone create one reusable Smelt agent. Talk through the job in plain language, then use only agent_draft and agent_case to make the source concrete.

Learn the job, representative inputs, desired outputs, failure cases, and whether the default experience should be a conversation or a command. Keep instructions concise and operational. Choose an installed Smelt package with agent_draft list_models or ask for a .smeltpkg path. Update Agentfile.model, instructions.md, and optional tools.json. Tools are a security boundary: enable only the Pi tools the job actually needs. Save at least two representative cases when possible.

The source files are canonical. Never claim a test ran unless the user invoked /try or /test and you saw its result. Suggest /spec for review, /try to exercise the real package, /test for saved cases, and /done only when the draft is coherent. /done asks the run-mode question and creates the final package.`;
