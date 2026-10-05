import assert from "node:assert/strict";
import { test } from "node:test";
async function moduleUnderTest(): Promise<any> {
  const name = "../src/pi-capture.js";
  const module = await import(name).catch(() => null);
  assert.ok(module, "Pi capture module exists");
  return module;
}
test("run ledger excludes baseline entries and keeps every new official entry", async () => {
  const { newSessionEntries } = await moduleUnderTest();
  const old = { id: "a", type: "message", message: { usage: { totalTokens: 9 } } };
  const fresh = { id: "b", parentId: "a", type: "message", message: { role: "assistant", usage: { output: 3, reasoning: 2, future: 99 } } };
  assert.deepEqual(newSessionEntries([old, fresh, fresh], new Set(["a"])), [fresh]);
});
test("bash recovery rejects traversal, foreign files, and symlinks before invoking container", async () => {
  const { isPiBashPath, bashReadScript } = await moduleUnderTest();
  assert.equal(isPiBashPath("/tmp/pi-bash-0123456789abcdef.log"), true);
  for (const value of ["/etc/passwd", "/tmp/../tmp/pi-bash-0123456789abcdef.log", "/tmp/pi-bash-secret.log"])
    assert.equal(isPiBashPath(value), false);
  const { spawnSync } = await import("node:child_process");
  const { symlinkSync, unlinkSync } = await import("node:fs");
  const file = "/tmp/pi-bash-fedcba9876543210.log";
  symlinkSync("/etc/passwd", file);
  try {
    const result = spawnSync(process.execPath, ["-e", bashReadScript, file]);
    assert.notEqual(result.status, 0);
    assert.equal(result.stdout.length, 0);
  }
  finally {
    unlinkSync(file);
  }
});

test("cursor replacement and recovery never follow container-created symlinks", async () => {
  const { PiRunCapture } = await moduleUnderTest();
  const fs = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const path = await import("node:path");
  const root = await fs.mkdtemp(path.join(tmpdir(), "capture-security-"));
  const session = path.join(root, "session");
  await fs.mkdir(session);
  const outside = path.join(root, "outside");
  const cursor = path.join(session, "capture-cursor.json");
  const events: any[] = [];
  const telemetry = { record: async (_r: string, _s: string, kind: string, payload: any) => { events.push({kind,payload}); return true; }, setPiSessionId() {}, markIncomplete() {}, completeness: () => [] };
  const capture = new PiRunCapture(telemetry, "r", "s", session, "container");
  try {
    await fs.writeFile(outside, "host sentinel");
    await fs.symlink(outside, cursor);
    await capture.beforePrompt({snapshot: async () => ({state:{sessionId:"pi"}, entries:{entries:[]}, stats:{}})});
    assert.equal(await fs.readFile(outside, "utf8"), "host sentinel");
    assert.equal((await fs.lstat(cursor)).isSymbolicLink(), true);
    const trustedCursor = path.join(`${session}.capture`, "r.json");
    assert.equal((await fs.lstat(trustedCursor)).isSymbolicLink(), false);
    await fs.writeFile(outside, JSON.stringify({type:"session",id:"pi"}) + '\n' + JSON.stringify({id:"private",type:"message",message:{content:[]}}) + '\n');
    await fs.symlink(outside, path.join(session,"attack.jsonl"));
    await fs.writeFile(path.join(session,"valid.jsonl"), JSON.stringify({type:"session",id:"pi"}) + '\n' + JSON.stringify({id:"new",type:"message",message:{content:[]}}) + '\n');
    await capture.recover();
    assert.deepEqual(events.filter(e => e.kind === "session_entry").map(e => e.payload.entry.id), ["new"]);
    events.length = 0;
    await fs.unlink(trustedCursor);
    await fs.writeFile(outside, JSON.stringify({run_id:"r",session_id:"s",pi_session_id:"pi",baseline:[]}));
    await fs.symlink(outside, trustedCursor);
    await capture.recover();
    assert.equal(events.filter(e => e.kind === "session_entry").length, 0);
  } finally { await fs.rm(root,{recursive:true,force:true}); }
});

test("capture consumes asynchronous telemetry failures", async () => {
  const { PiRunCapture } = await moduleUnderTest();
  const reasons: string[] = [];
  const capture = new PiRunCapture({record: async () => {throw new Error("writer failed");}, markIncomplete: (_id: string, reason: string) => reasons.push(reason)} as any, "r", "s", "/unused", "container");
  capture.onCommand({type:"prompt"});
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(reasons, ["capture_write_failed"]);
});

test("container cannot replace the trusted run baseline cursor", async () => {
  const {PiRunCapture} = await moduleUnderTest();
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const {tmpdir} = await import("node:os");
  const root = await fs.mkdtemp(path.join(tmpdir(),"capture-baseline-"));
  const session = path.join(root,"session");
  await fs.mkdir(session);
  const entries: any[] = [];
  const telemetry = {record:async (_r: string,_s: string,kind: string,payload: any)=>{if(kind==="session_entry") entries.push(payload.entry);return true;},markIncomplete(){},completeness:()=>[],setPiSessionId(){}};
  try {
    const capture = new PiRunCapture(telemetry,"r","s",session,"container");
    await capture.beforePrompt({snapshot:async()=>({state:{sessionId:"pi"},entries:{entries:[]},stats:{}})});
    await fs.writeFile(path.join(session,"capture-cursor.json"),JSON.stringify({run_id:"r",session_id:"s",pi_session_id:"pi",baseline:["fresh"]}));
    await fs.writeFile(path.join(session,"valid.jsonl"),JSON.stringify({type:"session",id:"pi"})+'\n'+JSON.stringify({id:"fresh",type:"message",message:{content:[]}})+'\n');
    await new PiRunCapture(telemetry,"r","s",session,"stopped").recover();
    assert.deepEqual(entries.map(e=>e.id),["fresh"]);
  } finally {await fs.rm(root,{recursive:true,force:true});}
});

test("cancelled finalizers never record after a late snapshot resolves", async () => {
  const {PiRunCapture} = await moduleUnderTest();
  const recorded: string[] = [];
  const telemetry = {record:async (_r: string,_s: string,kind: string)=>{recorded.push(kind);return true;},markIncomplete(){},completeness:()=>[]};
  const capture = new PiRunCapture(telemetry,"r","s","/unused","container");
  let resolve!: (value: any)=>void;
  const done = capture.finalize({snapshot:()=>new Promise(r=>{resolve=r;})});
  capture.cancel();
  resolve({state:{},stats:{},entries:{entries:[]}});
  await done;
  capture.onRecord({type:"late"});
  capture.onCommand({type:"late"});
  await new Promise(r=>setImmediate(r));
  assert.deepEqual(recorded,[]);
});

test("recovery bounds directory enumeration including non-session files", async () => {
  const {PiRunCapture} = await moduleUnderTest();
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const {tmpdir} = await import("node:os");
  const root = await fs.mkdtemp(path.join(tmpdir(),"capture-enumeration-"));
  const session = path.join(root,"session");
  await fs.mkdir(session);
  const reasons: string[] = [];
  const telemetry = {record:async()=>true,markIncomplete:(_id: string,reason: string)=>reasons.push(reason),completeness:()=>reasons,setPiSessionId(){}};
  try {
    const capture = new PiRunCapture(telemetry,"r","s",session,"container");
    await capture.beforePrompt({snapshot:async()=>({state:{sessionId:"pi"},entries:{entries:[]},stats:{}})});
    await Promise.all(Array.from({length:130},(_,i)=>fs.writeFile(path.join(session,`noise-${i}`),"")));
    await capture.recover();
    assert.ok(reasons.includes("session_recovery_failed"));
  } finally {await fs.rm(root,{recursive:true,force:true});}
});
