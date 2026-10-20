import assert from "node:assert/strict";
import type { Server } from "node:http";
import { test } from "node:test";

import { RuntimeConflictError, RuntimeNotFoundError } from "../src/orchestrator.js";
import { EventEmitter } from "node:events";
import { authorized, createAgentHttpServer, guardProcess, parseJevClassifierMode } from "../src/server.js";

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
    prompt: async (id: string, prompt: string, user: string, ...options: unknown[]) => { calls.push(["prompt", id, prompt, user, ...options]); return { session_id: id, run_id: "R2", status: "starting" }; },
    getRun: (id: string) => { if (id === "missing") throw new RuntimeNotFoundError(); return { ...run, run_id: id }; },
    listEvents: () => run.events,
    decide: (...args: any[]) => calls.push(["decide", ...args]),
    cancel: async (...args: any[]) => calls.push(["cancel", ...args]),
    closeSession: async (...args: any[]) => calls.push(["close", ...args]),
    tasks: (id: string) => [{ id: 'inspect',workflow: id,state: 'succeeded' }],
    resume: async (id: string, user: string) => { calls.push(['resume',id,user]); return { run_id: id,session_id: 'S1',status: 'starting' }; },
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

test('workflow task and resume routes use the existing auth contract and validate prompt modes', async () => {
  const { server,calls } = fixture(); const base = await listen(server);
  const headers = { authorization: 'Bearer runtime-secret','content-type': 'application/json' };
  try {
    assert.equal((await fetch(`${base}/v1/runs/R1/tasks`)).status,401);
    const tasks = await fetch(`${base}/v1/runs/R1/tasks`,{ headers });
    assert.deepEqual(await tasks.json(),[{ id: 'inspect',workflow: 'R1',state: 'succeeded' }]);
    const resumed = await fetch(`${base}/v1/runs/R1/resume`,{ method: 'POST',headers,body: JSON.stringify({ user_id: 'owner' }) });
    assert.equal(resumed.status,202); assert.deepEqual(calls.at(-1),['resume','R1','owner']);
    const invalid = await fetch(`${base}/v1/sessions/S1/prompts`,{ method: 'POST',headers,body: JSON.stringify({ user_id: 'owner',prompt: 'fix',workflow_mode: 'execute-js' }) });
    assert.equal(invalid.status,400);
  } finally { await close(server); }
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

test("feedback accepts a reason category and a bounded, redacted comment on any terminal status", async () => {
  let status = "timed_out"; const captures: any[] = [];
  const server=createAgentHttpServer({token:"secret",health:()=>({}),orchestrator:{getRun:()=>({run_id:"r",session_id:"s",owner_user_id:"owner",status})} as any,
    feedback: async (_run:any,payload:any,id:string)=>{captures.push({payload,id});return true;}} as any);
  const base=await listen(server);
  const submit=(body:Record<string,unknown>)=>fetch(`${base}/v1/runs/r/feedback`,{method:"POST",headers:{authorization:"Bearer secret","content-type":"application/json"},
    body:JSON.stringify({user_id:"owner",verdict:"needs_changes",feedback_id:"click-2",...body})});
  try {
    assert.equal((await submit({reason_category:"vibes"})).status,400);
    assert.equal((await submit({reason_category:7})).status,400);
    assert.equal((await submit({comment:"x".repeat(2001)})).status,400);
    assert.equal((await submit({comment:42})).status,400);
    assert.equal((await submit({user_id:"other",reason_category:"style"})).status,401);
    assert.equal(captures.length,0);
    const comment="please retry\nOPENAI_API_KEY=sk-live-should-not-leak\nAuthorization: Bearer abc.def-ghi";
    assert.equal((await submit({reason_category:"broke_tests",comment})).status,202);
    assert.equal(captures[0].payload.reason_category,"broke_tests");
    assert.match(captures[0].payload.comment,/please retry/);
    assert.doesNotMatch(captures[0].payload.comment,/sk-live-should-not-leak|abc\.def-ghi/);
    assert.match(captures[0].payload.comment,/\[REDACTED\]/);
    assert.equal((await submit({reason_category:"broke_tests",comment})).status,202);
    assert.equal(captures[1].id,captures[0].id);
    for (const terminal of ["completed","failed","cancelled","rejected","interrupted"]) {
      status=terminal;
      assert.equal((await submit({reason_category:"other",comment:"x".repeat(2000),feedback_id:`click-${terminal}`})).status,202);
    }
    status="completed";
    assert.equal((await submit({verdict:"accepted",feedback_id:"click-3"})).status,202);
    assert.equal("reason_category" in captures.at(-1).payload,false);
    assert.equal("comment" in captures.at(-1).payload,false);
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


test('prompt task mode is validated and forwarded separately from workflow mode', async () => {
  const { server,calls } = fixture(); const base = await listen(server);
  const headers = { authorization: 'Bearer runtime-secret','content-type': 'application/json' };
  try {
    for (const mode of ['discuss','implement']) {
      const response = await fetch(`${base}/v1/sessions/S1/prompts`,{ method: 'POST',headers,body: JSON.stringify({ user_id: 'U',prompt: 'project',task_mode: mode }) });
      assert.equal(response.status,202); assert.equal(calls.at(-1).at(-1),mode);
    }
    for (const mode of ['auto',null,{}]) {
      const before = calls.length;
      const response = await fetch(`${base}/v1/sessions/S1/prompts`,{ method: 'POST',headers,body: JSON.stringify({ user_id: 'U',prompt: 'project',task_mode: mode }) });
      assert.equal(response.status,400); assert.equal(calls.length,before);
    }
  } finally { await close(server); }
});

test('CI repair rejects discussion task mode before creating a repair', async () => {
  let created = false;
  const server = createAgentHttpServer({ token: 'secret',orchestrator: {} as any,health: () => ({}),
    ci: { checkContext: () => {},create: async () => { created = true; return {}; } } as any });
  const base = await listen(server);
  try {
    const response = await fetch(`${base}/v1/ci-repairs`,{ method: 'POST',headers: { authorization: 'Bearer secret','content-type': 'application/json' },body: JSON.stringify({ target: '7',user_id: 'U',task_mode: 'discuss' }) });
    assert.equal(response.status,400); assert.equal(created,false);
  } finally { await close(server); }
});

test("bearer auth compares tokens in constant time and rejects length or prefix mismatches", () => {
  const request = (authorization?: string | string[]) => ({ headers: authorization === undefined ? {} : { authorization } }) as any;
  assert.equal(authorized(request("Bearer runtime-secret"), "runtime-secret"), true);
  for (const header of [undefined, "", "Bearer runtime-secreT", "Bearer runtime-secret ", "Bearer runtime", "bearer runtime-secret", "runtime-secret", ["Bearer runtime-secret"]]) {
    assert.equal(authorized(request(header as any), "runtime-secret"), false, String(header));
  }
  assert.equal(authorized(request("Bearer "), ""), false);
});

test("agent limits parse positive integers and fall back to defaults", async () => {
  const { parseAgentLimits } = await import("../src/server.js");
  assert.deepEqual(parseAgentLimits({}), { maxTurns: 20, maxToolCalls: 40 });
  assert.deepEqual(parseAgentLimits({ PI_AGENT_MAX_TURNS: "3", PI_AGENT_MAX_TOOL_CALLS: "7" }), { maxTurns: 3, maxToolCalls: 7 });
  assert.throws(() => parseAgentLimits({ PI_AGENT_MAX_TURNS: "0" }), /PI_AGENT_MAX_TURNS/);
  assert.throws(() => parseAgentLimits({ PI_AGENT_MAX_TOOL_CALLS: "lots" }), /PI_AGENT_MAX_TOOL_CALLS/);
});

test("writer context mirrors the trusted profile's paths, check names, rubric and delivery limits", async () => {
  const { writerContextFor } = await import("../src/server.js");
  const context = writerContextFor({ allowedPaths: ["src", "tests"], rubric: ["correct"], checks: [{ name: "pytest", argv: ["pytest"], cwd: ".", timeoutMs: 1 }] } as any);
  assert.deepEqual(context, { allowedPaths: ["src", "tests"], checks: ["pytest"], rubric: ["correct"], diffLimits: { maxFiles: 50, maxBytes: 1048576, maxSingleFileBytes: 262144 } });
});

test("run JSON reports a retained patch without exposing its host path", async () => {
  const server = createAgentHttpServer({
    orchestrator: { getRun: (id: string) => ({ run_id: id, status: "failed", patch_artifact: "/runtime/artifacts/R1/undelivered.patch" }) } as any,
    token: "runtime-secret",
    health: () => ({ status: "healthy" }) as any,
  });
  const base = await listen(server);
  const response = await fetch(`${base}/v1/runs/R1`, { headers: { authorization: "Bearer runtime-secret" } });
  const body = await response.json();
  const text = JSON.stringify(body);
  await close(server);
  assert.equal(body.patch_retained, true);
  assert.equal(text.includes("undelivered.patch"), false);
});

test("profiled writers test through run_checks, so test runners leave the default safe commands", async () => {
  const { parseSafeCommands } = await import("../src/server.js");
  const runners = ["npm test", "npm run test", "pytest", "python -m pytest"];
  const unprofiled = parseSafeCommands(undefined);
  for (const runner of runners) assert.ok(unprofiled.includes(runner), runner);
  const profiled = parseSafeCommands(undefined, { profiled: true });
  for (const runner of runners) assert.ok(!profiled.includes(runner), runner);
  assert.deepEqual(profiled, ["npm run lint", "npm run build"]);
  assert.deepEqual(parseSafeCommands("pytest || make lint", { profiled: true }), ["pytest", "make lint"], "an explicit PI_AGENT_SAFE_COMMANDS still wins");
});

test("an unhandled rejection is logged and marks the runtime unhealthy instead of crashing it", () => {
  const target = new EventEmitter();
  const failures: unknown[] = []; const logs: string[] = [];
  const uninstall = guardProcess(target, error => failures.push(error), line => logs.push(line));
  target.emit("unhandledRejection", new Error("disk full"), Promise.resolve());
  assert.equal(failures.length, 1);
  assert.match(logs[0] ?? "", /"event":"agent_runtime_unhandled_rejection"/);
  assert.ok(!(logs[0] ?? "").includes("disk full"), "the reason text is not logged verbatim");
  uninstall();
  assert.equal(target.listenerCount("unhandledRejection"), 0);
});

test("POST /v1/sessions and prompts honour Idempotency-Key", async () => {
  const { IdempotencyCache } = await import("../src/idempotency.js");
  const calls: string[] = [];
  const orchestrator = {
    createSession: async () => { calls.push("create"); return { session_id: "S1", run_id: `R${calls.length}`, status: "starting" }; },
    prompt: async () => { calls.push("prompt"); return { session_id: "S1", run_id: `R${calls.length}`, status: "starting" }; },
  };
  const server = createAgentHttpServer({ orchestrator: orchestrator as any, token: "runtime-secret", health: () => ({}), idempotency: new IdempotencyCache() });
  const base = await listen(server);
  const post = (path: string, body: object, key?: string) => fetch(`${base}${path}`, { method: "POST", body: JSON.stringify(body),
    headers: { authorization: "Bearer runtime-secret", "content-type": "application/json", ...(key ? { "idempotency-key": key } : {}) } });
  const session = { team_id: "T", channel_id: "C", thread_ts: "1", user_id: "U", prompt: "fix" };
  try {
    const first = await (await post("/v1/sessions", session, "create-1")).json();
    const retry = await post("/v1/sessions", session, "create-1");
    assert.equal(retry.status, 202);
    assert.deepEqual(await retry.json(), first);
    assert.deepEqual(calls, ["create"]);
    const reused = await post("/v1/sessions", { ...session, prompt: "something else" }, "create-1");
    assert.equal(reused.status, 422);
    assert.equal((await reused.json()).error.code, "idempotency_key_reused");
    await post("/v1/sessions/S1/prompts", { prompt: "go", user_id: "U" }, "prompt-1");
    await post("/v1/sessions/S1/prompts", { prompt: "go", user_id: "U" }, "prompt-1");
    await post("/v1/sessions", session);
    assert.deepEqual(calls, ["create", "prompt", "create"], "no key means no deduplication");
  } finally { await close(server); }
});
