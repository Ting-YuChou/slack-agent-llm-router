import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
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
