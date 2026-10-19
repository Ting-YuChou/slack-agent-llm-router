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

type Block = { type: string; text?: string };
const budgetText = (result: unknown) => ((result as { content?: Block[] } | undefined)?.content ?? []).filter((block) => block.text?.startsWith("[harness]")).map((block) => block.text!);

/** Runs one tool call and returns the harness notices appended to its result. */
async function callTool(policy: Awaited<ReturnType<typeof loadPolicy>>, ctx: unknown, content: Block[] = [{ type: "text", text: "ok" }]) {
  await policy.fire("tool_call", { toolName: "ls", input: { path: "." } }, ctx);
  return policy.fire("tool_result", { type: "tool_result", toolCallId: "t", toolName: "ls", input: {}, content, isError: false }, ctx);
}

test("tool results warn once at 75% of the tool-call budget and finally with two calls left", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "40", PI_AGENT_MAX_TURNS: "100" });
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  const notices: Array<[number, string[]]> = [];
  for (let call = 1; call <= 40; call++) {
    const text = budgetText(await callTool(policy, ctx));
    if (text.length) notices.push([call, text]);
  }
  assert.deepEqual(notices.map(([call]) => call), [30, 38]);
  assert.equal(notices[0][1].length, 1);
  assert.match(notices[0][1][0], /^\[harness\] Budget notice: 30\/40 tool calls used/);
  assert.match(notices[0][1][0], /wrap up/i);
  assert.match(notices[0][1][0], /test/);
  assert.match(notices[0][1][0], /summar/i);
  assert.match(notices[1][1][0], /^\[harness\] Final budget warning: 38\/40 tool calls used, 2 left/);
}));

test("turn budget warnings ride on tool results at 75% and with two turns left", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TURNS: "12", PI_AGENT_MAX_TOOL_CALLS: "1000" });
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  const notices: Array<[number, string[]]> = [];
  for (let turn = 1; turn <= 12; turn++) {
    await policy.fire("turn_start", {}, ctx);
    for (let call = 0; call < 2; call++) {
      const text = budgetText(await callTool(policy, ctx));
      if (text.length) notices.push([turn, text]);
    }
  }
  assert.deepEqual(notices.map(([turn]) => turn), [9, 10]);
  assert.match(notices[0][1][0], /^\[harness\] Budget notice: 9\/12 turns used/);
  assert.match(notices[1][1][0], /^\[harness\] Final budget warning: 10\/12 turns used, 2 left/);
}));

test("a small budget skips straight to the final warning instead of warning twice", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "4", PI_AGENT_MAX_TURNS: "100" });
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  const notices: string[][] = [];
  for (let call = 1; call <= 4; call++) notices.push(budgetText(await callTool(policy, ctx)));
  assert.deepEqual(notices.map((text) => text.length), [0, 1, 0, 0]);
  assert.match(notices[1][0], /Final budget warning: 2\/4 tool calls used, 2 left/);
}));

test("budget notices keep content other handlers added and nested results get none", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "4", PI_AGENT_MAX_TURNS: "100" });
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  await policy.fire("input", {}, ctx);
  await callTool(policy, ctx);
  await policy.fire("tool_call", { toolName: "ls", input: { path: "." } }, ctx);
  const nested = await policy.fire("tool_result", { type: "tool_result", toolCallId: "c/1", parentToolCallId: "c", toolName: "ls", input: {}, content: [{ type: "text", text: "ok" }], isError: false }, ctx);
  assert.equal(nested, undefined);
  const content = [{ type: "text", text: "tool output" }, { type: "text", text: "[checks] added by another handler" }];
  const structuredContent = { files: ["a"] };
  const result = await policy.fire("tool_result", { type: "tool_result", toolCallId: "t3", toolName: "ls", input: {}, content, structuredContent, isError: false }, ctx) as { content: Block[]; structuredContent: unknown };
  assert.deepEqual(result.content.slice(0, 2), content);
  assert.equal(result.content.length, 3);
  assert.match(result.content[2].text!, /Final budget warning: 2\/4 tool calls used/);
  assert.deepEqual(result.structuredContent, structuredContent);
  assert.equal(content.length, 2, "the event's own content array is not mutated");
}));

test("a new input resets the budget notices", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "8", PI_AGENT_MAX_TURNS: "100" });
  const ctx = { cwd, abort: () => undefined, ui: { confirm: async () => true } };
  const run = async () => {
    await policy.fire("input", {}, ctx);
    const calls: number[] = [];
    for (let call = 1; call <= 8; call++) if (budgetText(await callTool(policy, ctx)).length) calls.push(call);
    return calls;
  };
  assert.deepEqual(await run(), [6]);
  assert.deepEqual(await run(), [6]);
}));

/** A ctx whose host answers pi-agent/budget_status with the next reply; `asks` counts the dialogs. */
function budgetCtx(cwd: string, replies: Array<string | undefined>) {
  const asks: Array<[string, string | undefined]> = [];
  let next = 0;
  const ctx = { cwd, abort: () => undefined, ui: {
    confirm: async () => true,
    input: async (title: string, placeholder?: string) => { asks.push([title, placeholder]); return replies[Math.min(next++, replies.length - 1)]; },
  } };
  return { ctx, asks };
}

const usd = (used: number, limit = 10) => JSON.stringify({ used_usd: used, limit_usd: limit });

test("dollar budget: a notice at 75% and a final warning at 90%, once each", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "1000", PI_AGENT_MAX_TURNS: "100" });
  const { ctx, asks } = budgetCtx(cwd, [usd(1), usd(7.5), usd(8), usd(9), usd(9.5), usd(9.9)]);
  await policy.fire("input", {}, ctx);
  const notices: Array<[number, string[]]> = [];
  for (let turn = 1; turn <= 6; turn++) {
    await policy.fire("turn_start", {}, ctx);
    for (let call = 0; call < 2; call++) {
      const text = budgetText(await callTool(policy, ctx));
      if (text.length) notices.push([turn, text]);
    }
  }
  assert.deepEqual(notices.map(([turn]) => turn), [2, 4]);
  assert.match(notices[0][1][0], /^\[harness\] Budget notice: \$7\.50\/\$10\.00 of the model budget used/);
  assert.match(notices[1][1][0], /^\[harness\] Final budget warning: \$9\.00\/\$10\.00 of the model budget used, \$1\.00 left/);
  assert.equal(asks.length, 4, "one dialog per turn, and none after the final warning");
  assert.ok(asks.every(([title, placeholder]) => title === "pi-agent/budget_status" && placeholder === ""));
}));

test("dollar budget: an unavailable, cancelled or malformed reply adds no notice", () => withWorkspace(async (cwd) => {
  for (const reply of [JSON.stringify({ unavailable: true }), undefined, "not json", JSON.stringify({ used_usd: 9, limit_usd: 0 }), JSON.stringify({ used_usd: "9", limit_usd: 10 })]) {
    const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "1000", PI_AGENT_MAX_TURNS: "100" });
    const { ctx, asks } = budgetCtx(cwd, [reply]);
    await policy.fire("input", {}, ctx);
    await policy.fire("turn_start", {}, ctx);
    assert.deepEqual(budgetText(await callTool(policy, ctx)), [], String(reply));
    assert.equal(asks.length, 1);
  }
}));

test("dollar budget: a new input resets the dollar notices", () => withWorkspace(async (cwd) => {
  const policy = await loadPolicy({ PI_AGENT_MAX_TOOL_CALLS: "1000", PI_AGENT_MAX_TURNS: "100" });
  const { ctx } = budgetCtx(cwd, [usd(8)]);
  const run = async () => {
    await policy.fire("input", {}, ctx);
    const texts: string[] = [];
    for (let turn = 1; turn <= 3; turn++) { await policy.fire("turn_start", {}, ctx); texts.push(...budgetText(await callTool(policy, ctx))); }
    return texts.length;
  };
  assert.equal(await run(), 1);
  assert.equal(await run(), 1);
}));
