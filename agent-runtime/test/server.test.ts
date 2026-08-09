import assert from "node:assert/strict";
import type { Server } from "node:http";
import { test } from "node:test";

import { RuntimeConflictError, RuntimeNotFoundError } from "../src/orchestrator.js";
import { createAgentHttpServer } from "../src/server.js";

async function listen(server: Server): Promise<string> {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}
async function close(server: Server) {
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

function fixture() {
  const calls: any[] = [];
  const run = { run_id: "R1", session_id: "S1", status: "running", answer: "", events: [{ type: "status", status: "running" }] };
  const orchestrator = {
    createSession: async (body: any) => { calls.push(["create", body]); return { session_id: "S1", run_id: "R1", status: "starting" }; },
    lookupSession: () => ({ session_id: "S1", owner_user_id: "U1", active: false }),
    prompt: async (id: string, prompt: string, user: string) => { calls.push(["prompt", id, prompt, user]); return { session_id: id, run_id: "R2", status: "starting" }; },
    getRun: (id: string) => { if (id === "missing") throw new RuntimeNotFoundError(); return { ...run, run_id: id }; },
    listEvents: () => run.events,
    decide: (...args: any[]) => calls.push(["decide", ...args]),
    cancel: async (...args: any[]) => calls.push(["cancel", ...args]),
    closeSession: async (...args: any[]) => calls.push(["close", ...args]),
    getTree: async (...args: any[]) => { calls.push(["tree", ...args]); return { session_id: "S1", active_leaf_id: "leaf", turns: [], lineage: {}, truncated: false }; },
    getStats: async (...args: any[]) => { calls.push(["stats", ...args]); return { session_id: "S1", total_messages: 2, auto_compaction_enabled: true, compaction_count: 0 }; },
    compact: async (...args: any[]) => { calls.push(["compact", ...args]); return { session_id: "S1", run_id: "RC", status: "starting" }; },
    fork: async (...args: any[]) => { calls.push(["fork", ...args]); return { session_id: "S2", branch: "pi-agent/child", baseline_commit: "abc" }; },
  };
  const server = createAgentHttpServer({
    orchestrator: orchestrator as any,
    token: "runtime-secret",
    health: () => ({ status: "healthy", runtime: "pi-coding-agent", model: "gpt-5.6-luna", reasoning_effort: "max", tools: ["read", "write", "edit", "bash", "grep", "find", "ls"] }),
  });
  return { server, calls };
}

test("health is public while every session/run endpoint requires bearer auth", async () => {
  const { server } = fixture();
  const base = await listen(server);
  const health = await fetch(`${base}/health`);
  const denied = await fetch(`${base}/v1/runs/R1`);
  await close(server);

  assert.equal(health.status, 200);
  assert.equal((await health.json()).runtime, "pi-coding-agent");
  assert.equal(denied.status, 401);
  assert.equal((await denied.json()).error.code, "unauthorized");
});

test("session, prompt, decision, cancel, close, run and SSE endpoints follow async contract", async () => {
  const { server, calls } = fixture();
  const base = await listen(server);
  const headers = { authorization: "Bearer runtime-secret", "content-type": "application/json" };
  const created = await fetch(`${base}/v1/sessions`, { method: "POST", headers, body: JSON.stringify({ team_id: "T1", channel_id: "C1", thread_ts: "1", user_id: "U1", prompt: "fix" }) });
  const lookup = await fetch(`${base}/v1/sessions/lookup?team_id=T1&channel_id=C1&thread_ts=1`, { headers });
  const prompted = await fetch(`${base}/v1/sessions/S1/prompts`, { method: "POST", headers, body: JSON.stringify({ prompt: "test", user_id: "U1" }) });
  const status = await fetch(`${base}/v1/runs/R1`, { headers });
  const events = await fetch(`${base}/v1/runs/R1/events`, { headers });
  const decision = await fetch(`${base}/v1/runs/R1/decisions`, { method: "POST", headers, body: JSON.stringify({ approval_id: "A1", user_id: "U1", decision: "approve" }) });
  const cancel = await fetch(`${base}/v1/runs/R1/cancel`, { method: "POST", headers, body: JSON.stringify({ user_id: "U1" }) });
  const closed = await fetch(`${base}/v1/sessions/S1/close`, { method: "POST", headers, body: JSON.stringify({ user_id: "U1" }) });
  await close(server);

  assert.equal(created.status, 202);
  assert.equal((await lookup.json()).found, true);
  assert.equal(prompted.status, 202);
  assert.equal(status.status, 200);
  assert.match(events.headers.get("content-type") ?? "", /text\/event-stream/);
  assert.match(await events.text(), /event: status/);
  assert.equal(decision.status, 202);
  assert.equal(cancel.status, 202);
  assert.equal(closed.status, 202);
  assert.equal(calls.length, 5);
});

test("stable errors are sanitized and preserve 409/404 mapping", async () => {
  const { server } = fixture();
  const base = await listen(server);
  const headers = { authorization: "Bearer runtime-secret", "content-type": "application/json" };
  const missing = await fetch(`${base}/v1/runs/missing`, { headers });
  const invalid = await fetch(`${base}/v1/sessions`, { method: "POST", headers, body: "not-json" });
  await close(server);

  assert.equal(missing.status, 404);
  assert.equal((await missing.json()).error.code, "not_found");
  assert.equal(invalid.status, 400);
  assert.doesNotMatch(await invalid.text(), /stack|syntaxerror/i);
});

test("tree, stats, compaction, and fork endpoints preserve owner identity and async status", async () => {
  const { server, calls } = fixture();
  const base = await listen(server);
  const headers = { authorization: "Bearer runtime-secret", "content-type": "application/json" };
  const tree = await fetch(`${base}/v1/sessions/S1/tree?user_id=U1`, { headers });
  const stats = await fetch(`${base}/v1/sessions/S1/stats?user_id=U1`, { headers });
  const compact = await fetch(`${base}/v1/sessions/S1/compact`, {
    method: "POST", headers, body: JSON.stringify({ user_id: "U1", custom_instructions: "Preserve decisions" }),
  });
  const fork = await fetch(`${base}/v1/sessions/S1/forks`, {
    method: "POST", headers, body: JSON.stringify({ user_id: "U1", source_run_id: "R1", team_id: "T1", channel_id: "C1", thread_ts: "2" }),
  });
  await close(server);

  assert.equal(tree.status, 200);
  assert.equal(stats.status, 200);
  assert.equal(compact.status, 202);
  assert.equal(fork.status, 202);
  assert.deepEqual(calls.slice(-4), [
    ["tree", "S1", "U1"],
    ["stats", "S1", "U1"],
    ["compact", "S1", "U1", "Preserve decisions"],
    ["fork", "S1", { user_id: "U1", source_run_id: "R1", team_id: "T1", channel_id: "C1", thread_ts: "2" }],
  ]);
});
