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

test("feedback is authenticated, owner checked, terminal only, and has stable replay identity", async () => {
  let status = "completed"; const captures: any[] = [];
  const server=createAgentHttpServer({token:"secret",health:()=>({}),orchestrator:{getRun:()=>({run_id:"r",session_id:"s",owner_user_id:"owner",status})} as any,
    feedback: async (_run:any,payload:any,id:string)=>{captures.push({payload,id});return true;}} as any);
  const base=await listen(server);
  const submit=(user:string,token="secret")=>fetch(`${base}/v1/runs/r/feedback`,{method:"POST",headers:{authorization:`Bearer ${token}`,"content-type":"application/json"},body:JSON.stringify({user_id:user,verdict:"accepted",feedback_id:"click-1"})});
  try {
    assert.equal((await submit("owner","wrong")).status,401);
    assert.equal((await submit("other")).status,401);
    status="running"; assert.equal((await submit("owner")).status,409);
    status="completed"; assert.equal((await submit("owner")).status,202); assert.equal((await submit("owner")).status,202);
    assert.equal(captures.length,2); assert.equal(captures[0].id,captures[1].id);
  } finally {await close(server);}
});

test("restart capture recovery skips previously recovered terminal runs", async () => {
  const module: any = await import("../src/server.js");
  assert.equal(typeof module.recoverPendingCaptures,"function");
  const runs: any[] = [{run_id:"fresh",status:"interrupted"},{run_id:"old",status:"interrupted",capture_complete:false},{run_id:"done",status:"completed"}];
  const recovered: string[] = [];
  const recover = async (run:any)=>{recovered.push(run.run_id);run.capture_complete=false;};
  await module.recoverPendingCaptures(runs,recover);
  await module.recoverPendingCaptures(runs,recover);
  assert.deepEqual(recovered,["fresh"]);
});

test("failed terminal writes persist failure before releasing even if persistence fails", async () => {
  const module: any = await import("../src/server.js");
  assert.equal(typeof module.settleCaptureWrite,"function");
  const calls: string[] = [];
  const telemetry = {releaseRun:(id:string)=>calls.push(`release:${id}`)};
  await module.settleCaptureWrite("r",false,telemetry,async()=>{calls.push("persist:false");});
  assert.deepEqual(calls,["persist:false","release:r"]);
  calls.length=0;
  await assert.rejects(module.settleCaptureWrite("r",false,telemetry,async()=>{throw new Error("disk full");}));
  assert.deepEqual(calls,["release:r"]);
});
