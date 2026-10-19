import assert from "node:assert/strict";
import { test } from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

/** Loads the writer policy extension with a fake Pi API; MAX_* limits are read at import time. */
export async function loadPolicy(env: Record<string, string>) {
  const previous = Object.fromEntries(Object.keys(env).map((key) => [key, process.env[key]]));
  Object.assign(process.env, env);
  try {
    const extension = await import(`${pathToFileURL(path.resolve(import.meta.dirname, "../../extensions/policy.ts")).href}?${Math.random()}`);
    const handlers = new Map<string, Array<(...args: any[]) => any>>();
    const log: string[] = [];
    const entries: Array<{ customType: string; data: unknown }> = [];
    extension.default({
      on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, [...(handlers.get(event) ?? []), handler]),
      appendEntry: (customType: string, data: unknown) => { log.push(`entry:${customType}`); entries.push({ customType, data }); },
      registerTool: () => undefined,
    });
    const fire = async (event: string, payload: unknown, ctx: unknown) => {
      let result: unknown;
      for (const handler of handlers.get(event) ?? []) result = (await handler(payload, ctx)) ?? result;
      return result;
    };
    return { fire, log, entries };
  } finally {
    for (const [key, value] of Object.entries(previous)) if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
}

async function withWorkspace(run: (cwd: string) => Promise<void>) {
  const cwd = await mkdtemp(path.join(os.tmpdir(), "policy-ext-"));
  try { await run(cwd); } finally { await rm(cwd, { recursive: true, force: true }); }
}

test("tool-call limit records which limit was hit before aborting", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "1" });
  const ctx = { cwd, abort: () => policy.log.push("abort"), ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  assert.equal(await policy.fire("tool_call", { toolName: "ls", input: { path: "." } }, ctx), undefined);
  const blocked = await policy.fire("tool_call", { toolName: "ls", input: { path: "." } }, ctx) as { block: boolean };
  assert.equal(blocked.block, true);
  assert.deepEqual(policy.log, ["entry:pi_agent_limit", "abort"]);
  assert.deepEqual(policy.entries[0], { customType: "pi_agent_limit", data: { kind: "tool_calls", limit: 1 } });
}));

test("turn limit records which limit was hit before aborting", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TURNS: "1" });
  const ctx = { cwd, abort: () => policy.log.push("abort") };
  await policy.fire("input", {}, ctx);
  await policy.fire("turn_start", {}, ctx);
  assert.deepEqual(policy.log, []);
  await policy.fire("turn_start", {}, ctx);
  assert.deepEqual(policy.log, ["entry:pi_agent_limit", "abort"]);
  assert.deepEqual(policy.entries[0], { customType: "pi_agent_limit", data: { kind: "turns", limit: 1 } });
}));

const writerContext = { allowedPaths: ["src"], checks: ["pytest", "tsc"], rubric: ["tests_cover_change"], diffLimits: { maxFiles: 50, maxBytes: 1048576, maxSingleFileBytes: 262144 } };

test("writer is told its allowed paths, trusted checks and diff limits up front", async () => {
  const policy = await loadPolicy({ PI_AGENT_WRITER_JSON: JSON.stringify(writerContext) });
  const event = { systemPromptOptions: { sections: {} as Record<string, string> } };
  await policy.fire("before_agent_start", event, {});
  const scope = event.systemPromptOptions.sections.scope ?? "";
  for (const expected of ["src", "pytest", "tsc", "tests_cover_change", "50 files"]) assert.match(scope, new RegExp(expected));
});

test("writes outside the allowed paths are blocked at write time with an actionable reason", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_WRITER_JSON: JSON.stringify(writerContext) });
  let confirms = 0;
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => { confirms++; return true; } } };
  await policy.fire("input", {}, ctx);
  const blocked = await policy.fire("tool_call", { toolName: "write", input: { path: "docs/notes.md", content: "x" } }, ctx) as { block: boolean; reason: string };
  assert.equal(blocked.block, true);
  assert.match(blocked.reason, /outside_allowed_paths: docs\/notes\.md/);
  assert.match(blocked.reason, /src/);
  const edit = await policy.fire("tool_call", { toolName: "edit", input: { path: "../escape/src/a.ts", edits: [] } }, ctx) as { block: boolean };
  assert.equal(edit.block, true);
  assert.equal(await policy.fire("tool_call", { toolName: "write", input: { path: "src/a.ts", content: "x" } }, ctx), undefined);
  assert.equal(confirms, 1);
}));

test("without a writer context, writes are not path-scoped beyond workspace policy", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({});
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  assert.equal(await policy.fire("tool_call", { toolName: "write", input: { path: "docs/notes.md", content: "x" } }, ctx), undefined);
}));

/** The safe-command list is read per call, so it must stay in the environment while the test runs. */
async function withSafeCommands(commands: string[], run: () => Promise<void>) {
  const previous = process.env.PI_AGENT_SAFE_COMMANDS_JSON;
  process.env.PI_AGENT_SAFE_COMMANDS_JSON = JSON.stringify(commands);
  try { await run(); } finally {
    if (previous === undefined) delete process.env.PI_AGENT_SAFE_COMMANDS_JSON; else process.env.PI_AGENT_SAFE_COMMANDS_JSON = previous;
  }
}

async function withRepo(run: (cwd: string) => Promise<void>) {
  await withWorkspace(async (cwd) => {
    execFileSync("git", ["init", "-q"], { cwd });
    await writeFile(path.join(cwd, "README.md"), "x");
    execFileSync("git", ["add", "."], { cwd });
    execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init"], { cwd });
    await run(cwd);
  });
}

function confirmingCtx(cwd: string) {
  const prompts: string[] = [];
  return { prompts, ctx: { cwd, abort: () => undefined, ui: { confirm: async (_title: string, body: string) => { prompts.push(body); return true; } } } };
}

test("configured commands run without a prompt only while no test configuration has changed", () => withRepo((cwd) => withSafeCommands(["pytest"], async () => {
  const policy = await loadPolicy({});
  const { prompts, ctx } = confirmingCtx(cwd);
  await policy.fire("input", {}, ctx);
  assert.equal(await policy.fire("tool_call", { toolName: "write", input: { path: "src/a.py", content: "x" } }, ctx), undefined);
  assert.equal(prompts.length, 1);
  assert.equal(await policy.fire("tool_call", { toolName: "bash", input: { command: "pytest -q" } }, ctx), undefined);
  assert.equal(prompts.length, 1);

  await writeFile(path.join(cwd, "conftest.py"), "import os");
  assert.equal(await policy.fire("tool_call", { toolName: "bash", input: { command: "pytest -q" } }, ctx), undefined);
  assert.equal(prompts.length, 2);
  assert.match(prompts[1]!, /test configuration changed: conftest\.py/);
})));

test("a test configuration write in this session taints configured commands", () => withRepo((cwd) => withSafeCommands(["pytest"], async () => {
  const policy = await loadPolicy({});
  const { prompts, ctx } = confirmingCtx(cwd);
  await policy.fire("input", {}, ctx);
  await policy.fire("tool_call", { toolName: "write", input: { path: "pyproject.toml", content: "x" } }, ctx);
  await policy.fire("tool_call", { toolName: "bash", input: { command: "pytest" } }, ctx);
  assert.equal(prompts.length, 2);
  assert.match(prompts[1]!, /pyproject\.toml/);
})));

test("configured commands need approval when git cannot list changed files", () => withWorkspace((cwd) => withSafeCommands(["pytest"], async () => {
  const policy = await loadPolicy({});
  const { prompts, ctx } = confirmingCtx(cwd);
  await policy.fire("input", {}, ctx);
  await policy.fire("tool_call", { toolName: "write", input: { path: "src/a.py", content: "x" } }, ctx);
  await policy.fire("tool_call", { toolName: "bash", input: { command: "pytest" } }, ctx);
  assert.equal(prompts.length, 2);
  assert.match(prompts[1]!, /could not list changed files/);
})));

test("write approval mode none skips write prompts but keeps scope, protection and size limits", () => withRepo((cwd) => withSafeCommands(["pytest"], async () => {
  const policy = await loadPolicy({ PI_AGENT_WRITE_APPROVAL: "none", PI_AGENT_WRITER_JSON: JSON.stringify(writerContext) });
  const { prompts, ctx } = confirmingCtx(cwd);
  await policy.fire("input", {}, ctx);
  assert.equal(await policy.fire("tool_call", { toolName: "write", input: { path: "src/a.py", content: "x" } }, ctx), undefined);
  assert.equal(await policy.fire("tool_call", { toolName: "edit", input: { path: "src/a.py", edits: [] } }, ctx), undefined);
  assert.equal(await policy.fire("tool_call", { toolName: "bash", input: { command: "pytest -q" } }, ctx), undefined);
  assert.deepEqual(prompts, []);
  for (const input of [
    { path: "docs/a.md", content: "x" },
    { path: "src/.env", content: "x" },
    { path: "src/big.txt", content: "x".repeat(256 * 1024 + 1) },
  ]) {
    const blocked = await policy.fire("tool_call", { toolName: "write", input }, ctx) as { block?: boolean } | undefined;
    assert.equal(blocked?.block, true, input.path);
  }
  assert.equal(await policy.fire("tool_call", { toolName: "bash", input: { command: "python script.py" } }, ctx), undefined);
  assert.equal(prompts.length, 1, "unlisted shell commands still prompt");
})));

test("an unset or unknown write approval mode keeps per-write prompts", () => withWorkspace(async (cwd) => {
  for (const env of [{}, { PI_AGENT_WRITE_APPROVAL: "nope" }] as Array<Record<string, string>>) {
    const policy = await loadPolicy(env);
    const { prompts, ctx } = confirmingCtx(cwd);
    await policy.fire("input", {}, ctx);
    await policy.fire("tool_call", { toolName: "write", input: { path: "a.txt", content: "x" } }, ctx);
    assert.equal(prompts.length, 1, JSON.stringify(env));
  }
}));
