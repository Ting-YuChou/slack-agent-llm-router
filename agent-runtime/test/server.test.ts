import assert from "node:assert/strict";
import type { Server } from "node:http";
import { test } from "node:test";

import { RuntimeConflictError, RuntimeNotFoundError } from "../src/orchestrator.js";
import { createAgentHttpServer, parseJevClassifierMode } from "../src/server.js";

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
  };
  const server = createAgentHttpServer({
    orchestrator: orchestrator as any,
    token: "runtime-secret",
    health: () => ({ status: "healthy", runtime: "pi-coding-agent", model: "gpt-5.6-luna", reasoning_effort: "max", tools: ["read", "write", "edit", "bash", "grep", "find", "ls"] }),
  });
  return { server, calls };
}

test("run-time Jev classifier mode is explicit and defaults off", () => {
  assert.equal(parseJevClassifierMode(undefined), "off");
  assert.equal(parseJevClassifierMode(""), "off");
  assert.equal(parseJevClassifierMode("off"), "off");
  assert.equal(parseJevClassifierMode("on"), "on");
  assert.throws(() => parseJevClassifierMode("shadow"), /off or on/i);
});

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
