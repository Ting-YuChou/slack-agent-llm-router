import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { test } from "node:test";

import { buildAgentDockerArgs } from "../src/container-runtime.js";
import { RUN_CHECKS_DIALOG, RUN_CHECKS_TOOL } from "../src/pi-rpc.js";
import { writerTools } from "../src/server.js";
import { validateProfile } from "../src/verification.js";

const context = { checks: ["lint", "unit"], defaultChecks: ["lint"], maxCalls: 3 };

async function loadChecks(env: string | undefined) {
  const previous = process.env.PI_AGENT_CHECKS_JSON;
  if (env === undefined) delete process.env.PI_AGENT_CHECKS_JSON; else process.env.PI_AGENT_CHECKS_JSON = env;
  try {
    const module = await import(`${pathToFileURL(path.resolve(import.meta.dirname, "../../extensions/checks.ts")).href}?${Math.random()}`);
    const tools: any[] = []; const handlers = new Map<string, (...args: any[]) => any>();
    module.default({ registerTool: (tool: unknown) => tools.push(tool), on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler) });
    return { module, tool: tools[0], handlers };
  } finally { if (previous === undefined) delete process.env.PI_AGENT_CHECKS_JSON; else process.env.PI_AGENT_CHECKS_JSON = previous; }
}

test("the run_checks extension asks the host through the run_checks dialog and passes its abort signal", async () => {
  const { module, tool } = await loadChecks(JSON.stringify(context));
  assert.equal(module.RUN_CHECKS_TOOL, RUN_CHECKS_TOOL);
  assert.equal(module.RUN_CHECKS_DIALOG, RUN_CHECKS_DIALOG);
  assert.equal(tool.name, RUN_CHECKS_TOOL);
  const asked: any[] = []; const signal = new AbortController().signal;
  const answer = JSON.stringify({ status: "failed", checks: [{ name: "unit", status: "failed" }], report: "tests/a.py:2 boom" });
  const result = await tool.execute("call-7", { checks: ["unit"] }, signal, undefined, { ui: { input: async (...args: unknown[]) => { asked.push(args); return answer; } } });
  assert.deepEqual(asked, [[RUN_CHECKS_DIALOG, JSON.stringify({ tool_call_id: "call-7", checks: ["unit"] }), { signal }]]);
  assert.equal(result.content[0].text, answer);
  assert.equal(result.details.status, "failed");
  assert.equal(result.isError, false);
  const dismissed = await tool.execute("call-8", {}, signal, undefined, { ui: { input: async () => undefined } });
  assert.equal(dismissed.isError, true);
  const refused = await tool.execute("call-9", {}, signal, undefined, { ui: { input: async () => JSON.stringify({ status: "rejected", reason: "max_calls", checks: [], report: "" }) } });
  assert.equal(refused.isError, true);
});

test("the run_checks extension tells the writer the checks and its call limit", async () => {
  const { handlers } = await loadChecks(JSON.stringify(context));
  const event = { systemPromptOptions: { sections: {} as Record<string, string> } };
  handlers.get("before_agent_start")!(event);
  assert.match(event.systemPromptOptions.sections.run_checks, /lint, unit/);
  assert.match(event.systemPromptOptions.sections.run_checks, /at most 3 calls/);
  await assert.rejects(loadChecks(undefined), /PI_AGENT_CHECKS_JSON/);
});

const profile = validateProfile({ id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"],
  checks: [{ name: "lint", argv: ["lint"], cwd: ".", timeoutMs: 1000 }, { name: "unit", argv: ["unit"], cwd: ".", timeoutMs: 1000 }], writerChecks: { defaultChecks: ["lint"], maxCalls: 3 } });

test("server wires run_checks into the writer only when a trusted profile is set", () => {
  const base = { extensionPaths: ["/opt/pi/extensions/policy.ts"], toolNames: ["read", "bash"] };
  assert.deepEqual(writerTools(undefined, base.extensionPaths, base.toolNames), base);
  assert.deepEqual(writerTools(profile, base.extensionPaths, base.toolNames), {
    extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/checks.ts"], toolNames: ["read", "bash", "run_checks"], checksContext: context,
  });
});

test("the checks context reaches the writer container but never a discussion", () => {
  const options = { name: "n", image: "img", network: "net", worktreePath: "/w", gitMetadataPath: "/g", sessionStatePath: "/s", gatewayUrl: "http://g", modelRef: "openai/gpt-5.6-luna",
    pluginPaths: [], skillPaths: [], ...writerTools(profile, ["/opt/pi/extensions/policy.ts"], ["read", "bash"]) };
  const args = buildAgentDockerArgs(options as any);
  assert.ok(args.includes(`PI_AGENT_CHECKS_JSON=${JSON.stringify(context)}`));
  assert.ok(args.join(" ").includes("-e /opt/pi/extensions/checks.ts"));
  assert.ok(args.includes("read,bash,run_checks"));
  const discuss = buildAgentDockerArgs({ ...options, taskMode: "discuss" } as any);
  assert.ok(!discuss.some(arg => arg.includes("PI_AGENT_CHECKS_JSON") || arg.includes("checks.ts") || arg.includes("run_checks")));
});
