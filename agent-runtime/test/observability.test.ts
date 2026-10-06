import assert from "node:assert/strict";
import { test } from "node:test";
import { PiRpcBridge, RpcJsonlDecoder } from "../src/pi-rpc.js";
import { issueGatewayToken, issueClassifierGatewayToken } from "../src/gateway-token.js";
import { issueMcpToken } from "../src/mcp-token.js";
test("internal RPC capture retains all official and future usage fields without exposing thinking publicly", () => {
  const records: unknown[] = [];
  const publicEvents: unknown[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => { }, onEvent: (e: unknown) => publicEvents.push(e), onRecord: (r: unknown) => records.push(r) } as any);
  const record = { type: "message_end", message: { role: "assistant", content: [{ type: "thinking", thinking: "private" }], usage: { input: 2, output: 3, reasoning: 1, cacheRead: 4, cacheWrite: 5, cacheWrite1h: 6, totalTokens: 20, future: 9, cost: { input: .1, output: .2, total: .3, future: .4 } } } };
  bridge.feed(JSON.stringify(record) + "\n");
  assert.deepEqual(records, [record]);
  assert.doesNotMatch(JSON.stringify(publicEvents), /private|future|cacheRead/);
});
test("split UTF8 bytes preserve full dialogue", () => {
  const decoder = new RpcJsonlDecoder();
  const bytes = Buffer.from('{"text":"中文"}\n');
  const split = bytes.indexOf(Buffer.from("中")) + 1;
  assert.deepEqual([...decoder.push(bytes.subarray(0, split)), ...decoder.push(bytes.subarray(split))], [{ text: "中文" }]);
});
test("capture failure does not interrupt Pi", () => {
  const events: unknown[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => { }, onEvent: (e: unknown) => events.push(e), onRecord: () => { throw new Error("disk full"); } } as any);
  assert.doesNotThrow(() => bridge.feed('{"type":"turn_start"}\n'));
  assert.deepEqual(events, [{ type: "turn" }]);
});
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
async function telemetryModule(): Promise<any> {
  const location = "../src/telemetry.js";
  const module = await import(location).catch(() => null);
  assert.ok(module, "durable telemetry module exists");
  return module;
}
test("outbox reopens unacknowledged data, uses stable ids, and chunks large content", async () => {
  const { TelemetryStore, contentRecords } = await telemetryModule();
  const root = await mkdtemp(path.join(tmpdir(), "agent-outbox-"));
  try {
    const file = path.join(root, "events.sqlite");
    let store = new TelemetryStore(file);
    const records = contentRecords({ run_id: "r", session_id: "s", event_id: "e" }, "dialogue", "中".repeat(200000));
    assert.ok(records.length > 1);
    for (const record of records) {
      assert.ok(Buffer.from(record.payload.data, "base64").length <= 262144);
      store.append(record);
    }
    store.close();
    store = new TelemetryStore(file);
    assert.equal(store.pending().length, records.length);
    assert.equal(store.health().bytes, records.reduce((sum: number, r: any) => sum + Buffer.byteLength(JSON.stringify(r)), 0));
    for (const record of records)
      store.append(record);
    assert.equal(store.pending().length, records.length);
    store.ack(records.map((r: any) => r.event_id));
    assert.equal(store.pending().length, 0);
    assert.equal(store.health().bytes, 0);
    store.close();
  }
  finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("redaction covers known credentials in dialogue while preserving unknown metric fields", async () => {
  const { redact } = await telemetryModule();
  assert.deepEqual(redact({ usage: { future: 3 }, text: "key=secret-123", authorization: "Bearer other" }, ["secret-123"]), { usage: { future: 3 }, text: "key=[REDACTED]", authorization: "[REDACTED]" });
});
test("recovered session text redacts old signed credentials without an in-memory secret registry", async () => {
  const { redact } = await telemetryModule();
  const gateway = issueGatewayToken({runId: 'old-run', provider: 'openai', model: 'gpt-5.6-luna', api: 'openai-responses', reasoningEffort: 'max', expiresAt: 1}, 'previous-secret');
  const classifier = issueClassifierGatewayToken({kind: 'classifier', runId: 'old-run', provider: 'openrouter', model: 'typesafe/jev-1.13', api: 'openrouter-jev', maxCalls: 1, expiresAt: 1} as any, 'previous-secret');
  const mcp = issueMcpToken({run: 'old-run', session: 's', slackUser: 'u', repository: 'o/r', server: 'github', mode: 'github_read_only', tools: ['read'], expiry: 1}, 'previous-secret');
  for(const token of [gateway, classifier, mcp])
    assert.equal(redact('KEY=' + token, []), 'KEY=[REDACTED]');
  assert.equal(redact('ordinary.versioned.text', []), 'ordinary.versioned.text');
});
test("feedback replay after ACK does not replace a later human correction", async () => {
  const { TelemetryStore } = await telemetryModule();
  const root = await mkdtemp(path.join(tmpdir(), 'agent-feedback-'));
  try {
    const file = path.join(root, 'outbox.sqlite');
    let store = new TelemetryStore(file);
    store.append({ event_id: 'click-a', kind: 'feedback', topic: 'agent.events.v1', run_id: 'r', payload: { verdict: 'accepted' } });
    store.ack(['click-a']);
    store.close();
    store = new TelemetryStore(file);
    store.append({ event_id: 'click-b', kind: 'feedback', topic: 'agent.events.v1', run_id: 'r', payload: { verdict: 'needs_changes' } });
    store.append({ event_id: 'click-a', kind: 'feedback', topic: 'agent.events.v1', run_id: 'r', payload: { verdict: 'accepted' } });
    assert.equal(store.pending().length, 1);
    assert.equal(store.pending()[0].event_id, 'click-b');
    store.close();
  }
  finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("snapshot commands capture their full responses before prompt execution", async () => {
  const raw: unknown[] = [];
  const writes: any[] = [];
  let bridge!: PiRpcBridge;
  bridge = new PiRpcBridge({
    onRecord: r => raw.push(r), onEvent: () => { }, writeLine: line => {
      const command = JSON.parse(line);
      writes.push(command);
      const data = command.type === 'get_state' ? { model: { provider: 'openai', id: 'gpt-5.6-luna' }, thinkingLevel: 'max', sessionId: 'pi' } : command.type === 'get_entries' ? { entries: [{ id: 'e', type: 'usage', usage: { future: 9 } }], leafId: 'e' } : { totalMessages: 4 };
      if (command.type !== 'prompt')
        queueMicrotask(() => bridge.feed(JSON.stringify({ type: 'response', id: command.id, command: command.type, success: true, data }) + '\n'));
    }
  });
  let snapshot: any;
  await bridge.configureAndPrompt('r', 'hello', 'openai', 'gpt-5.6-luna', 'max', async () => { snapshot = await bridge.snapshot(); });
  assert.equal(writes.at(-1).type, 'prompt');
  assert.equal(snapshot.entries.entries[0].usage.future, 9);
  assert.equal(raw.length, 6);
});
test("binary bash archive preserves invalid UTF8 and NUL while redacting credentials", async () => {
  const { redactBytes } = await telemetryModule();
  const bytes = Buffer.concat([Buffer.from([0, 255, 254]), Buffer.from('secret-中文'), Buffer.from([1, 128])]);
  assert.deepEqual(redactBytes(bytes, ['secret-中文']), Buffer.concat([Buffer.from([0, 255, 254]), Buffer.from('[REDACTED]'), Buffer.from([1, 128])]));
});
test("streaming binary redaction covers credentials across chunks without leaking a long bearer suffix", async () => {
  const { BinaryRedactor } = await telemetryModule();
  const secret = '密碼-' + 'x'.repeat(200);
  const input = Buffer.concat([Buffer.from([0, 255]), Buffer.from('secret=' + secret + '\nBearer ' + 'z'.repeat(4000) + '\nend'), Buffer.from([254])]);
  const redactor = new BinaryRedactor([secret]);
  const output: Buffer[] = [];
  for (let i = 0;i < input.length;i += 113)
    output.push(redactor.push(input.subarray(i, i + 113)));
  output.push(redactor.finish());
  assert.deepEqual(Buffer.concat(output), Buffer.concat([Buffer.from([0, 255]), Buffer.from('secret=[REDACTED]\nBearer [REDACTED]\nend'), Buffer.from([254])]));
});
test("capacity rejects new batches atomically and still accepts pending replay identities", async () => {
  const { TelemetryStore } = await telemetryModule();
  const root = await mkdtemp(path.join(tmpdir(), 'agent-capacity-'));
  try {
    const row = { event_id: 'a', run_id: 'r', kind: 'rpc', topic: 'agent.events.v1', payload: 'x'.repeat(100) };
    const bytes = Buffer.byteLength(JSON.stringify(row));
    const store = new TelemetryStore(path.join(root, 'outbox.sqlite'), bytes + 10);
    store.append(row);
    assert.doesNotThrow(() => store.append(row));
    assert.throws(() => store.appendMany([{ ...row, event_id: 'b' }, { ...row, event_id: 'c' }]), /capacity/);
    assert.deepEqual(store.pending().map((r: any) => r.event_id), ['a']);
    assert.equal(store.health().bytes, bytes);
    assert.equal(store.health().depth, 1);
    store.close();
  }
  finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("prompt and approval replies are captured even when no session entry is produced", () => {
  const commands: unknown[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => { }, onEvent: () => { }, onCommand: r => commands.push(r) });
  bridge.prompt('r', '完整 prompt');
  bridge.respondToUi('approval', false);
  assert.deepEqual(commands, [{ id: 'r', type: 'prompt', message: '完整 prompt' }, { type: 'extension_ui_response', id: 'approval', confirmed: false }]);
});
test("Kafka ACK uncertainty retains durable events and replays the same event identity", async () => {
  const { TelemetryStore, publishOutbox } = await telemetryModule();
  const root = await mkdtemp(path.join(tmpdir(), 'agent-ack-'));
  try {
    const store = new TelemetryStore(path.join(root, 'outbox.sqlite'));
    const event = { event_id: 'same-id', run_id: 'r', topic: 'agent.events.v1', kind: 'rpc', payload: { future: 9 } };
    store.append(event);
    const delivered: string[] = [];
    let fail = true;
    const producer = {
      send: async (request: any) => {
        assert.equal(request.acks, -1); delivered.push(JSON.parse(request.messages[0].value).event_id); if (fail)
          throw new Error('ACK response lost');
      }
    };
    assert.equal(typeof publishOutbox, 'function');
    await assert.rejects(() => publishOutbox(store, producer));
    assert.equal(store.pending().length, 1);
    fail = false;
    await publishOutbox(store, producer);
    assert.equal(store.pending().length, 0);
    assert.deepEqual(delivered, ['same-id', 'same-id']);
    store.close();
  }
  finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Kafka publishing keeps each produce request below the broker message budget", async () => {
  const { TelemetryStore, publishOutbox } = await telemetryModule();
  const root = await mkdtemp(path.join(tmpdir(), 'agent-kafka-batch-'));
  try {
    const store = new TelemetryStore(path.join(root, 'outbox.sqlite'));
    for (let i = 0; i < 3; i++)
      store.append({ event_id: `large-${i}`, run_id: 'r', topic: 'agent.content.v1', kind: 'content', payload: 'x'.repeat(350_000) });
    const sizes: number[] = [];
    await publishOutbox(store, { send: async (request: { messages: Array<{ value: string }> }) => {
      sizes.push(request.messages.reduce((total: number, message: { value: string }) => total + Buffer.byteLength(message.value), 0));
    } });
    assert.ok(sizes.length > 1);
    assert.ok(sizes.every(size => size <= 768 * 1024));
    assert.equal(store.pending().length, 0);
    store.close();
  }
  finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prefixed credential keys are redacted without deleting native token metrics", async () => {
  const {redact} = await telemetryModule();
  assert.deepEqual(redact({OPENAI_API_KEY:"hidden", DATABASE_PASSWORD:"hidden", GITHUB_TOKEN:"hidden", usage:{totalTokens:9,input_tokens:3}}), {OPENAI_API_KEY:"[REDACTED]",DATABASE_PASSWORD:"[REDACTED]",GITHUB_TOKEN:"[REDACTED]",usage:{totalTokens:9,input_tokens:3}});
});
test("deep JSON serialization failures return incomplete rather than reject", async () => {
  const {AgentTelemetry} = await telemetryModule();
  // Exercise the real record method without starting a Kafka worker.
  const telemetry = Object.create(AgentTelemetry.prototype);
  Object.assign(telemetry,{sequences:new Map(),versions:new Map(),runContext:new Map(),incomplete:new Map(),issuedSecrets:new Map(),secrets:[]});
  const deep = JSON.parse('{"a":'.repeat(20000) + '0' + '}'.repeat(20000));
  assert.equal(await telemetry.record("r","s","rpc",deep),false);
  assert.equal(telemetry.completeness("r").length,1);
});

test("streamed credential assignments redact unknown values across arbitrary chunks", async () => {
  const {BinaryRedactor,redact} = await telemetryModule();
  const text = 'OPENAI_API_KEY="' + 's'.repeat(1000) + '"\nDATABASE_PASSWORD=db-secret\nGITHUB_TOKEN=gh-secret\nusage=42\n';
  assert.doesNotMatch(redact(text), /db-secret|gh-secret|ssssssss/);
  for (const size of [1,17,300,10000]) {
    const redactor = new BinaryRedactor([]);
    const chunks: Buffer[] = [];
    for(let i=0;i<text.length;i+=size) chunks.push(redactor.push(Buffer.from(text.slice(i,i+size))));
    chunks.push(redactor.finish());
    const output = Buffer.concat(chunks).toString();
    assert.doesNotMatch(output,/db-secret|gh-secret|ssssssss/);
    assert.match(output,/usage=42/);
  }
});

test("terminal cleanup releases issued secrets and per-run state", async () => {
  const {AgentTelemetry} = await telemetryModule();
  const telemetry = Object.create(AgentTelemetry.prototype);
  Object.assign(telemetry,{sequences:new Map([["r",1]]),versions:new Map([["r",1n]]),runContext:new Map([["r",{}]]),incomplete:new Map([["r",new Set(["failed"])]]),issuedSecrets:new Map(),secrets:["static-secret"]});
  telemetry.addSecret("run-secret","r");
  assert.equal(telemetry.binaryRedactor("r").push(Buffer.from("" )).length,0);
  assert.equal(telemetry.binaryRedactor("r").finish().length,0);
  telemetry.releaseRun("r");
  assert.equal(telemetry.issuedSecrets.size,0);
  for (const name of ["sequences","versions","runContext","incomplete"]) assert.equal(telemetry[name].size,0);
  assert.equal(telemetry.redactText("static-secret"),"[REDACTED]");
});

test("streamed Basic authorization is masked and malformed quotes retain following lines", async () => {
  const {BinaryRedactor,redact} = await telemetryModule();
  const input = 'Authorization: Basic abcdef\nDATABASE_PASSWORD=my secret password\nOPENAI_API_KEY="short-secret\nOPENAI_API_KEY="'+'s'.repeat(1000)+'\nimportant test output\n';
  assert.doesNotMatch(redact(input),/abcdef|ssssssss|short-secret|my secret password/);
  for(const size of [1,2,7,256,4096]) {
    const redactor = new BinaryRedactor([]);
    const output: Buffer[] = [];
    for(let i=0;i<input.length;i+=size) output.push(redactor.push(Buffer.from(input.slice(i,i+size))));
    output.push(redactor.finish());
    const text = Buffer.concat(output).toString();
    assert.doesNotMatch(text,/abcdef|ssssssss|short-secret|my secret password/);
    assert.match(text,/important test output/);
  }
});

test("opaque JSON and escaped quoted credentials cannot bypass streaming redaction", async () => {
  const {BinaryRedactor,redact} = await telemetryModule();
  const input = '{"OPENAI_API_KEY":"abc\\"LEAKME"}\nOPENAI_API_KEY="abc\\"LEAKME"\nDATABASE_PASSWORD=[REDACTED]LEAKME\nnext line\n';
  assert.doesNotMatch(redact(input),/LEAKME/);
  for (const size of [1,7,256,4096]) {
    const redactor = new BinaryRedactor([]);
    const chunks: Buffer[] = [];
    for(let i=0;i<input.length;i+=size) chunks.push(redactor.push(Buffer.from(input.slice(i,i+size))));
    chunks.push(redactor.finish());
    const output = Buffer.concat(chunks).toString();
    assert.doesNotMatch(output,/LEAKME/);
    assert.match(output,/next line/);
  }
});

test("AWS and private-key credentials are masked in objects and opaque streams", async () => {
  const {redact,BinaryRedactor} = await telemetryModule();
  assert.deepEqual(redact({AWS_SECRET_ACCESS_KEY:"LEAKME",SSH_PRIVATE_KEY:"LEAKME",DJANGO_SECRET_KEY:"LEAKME"}),{AWS_SECRET_ACCESS_KEY:"[REDACTED]",SSH_PRIVATE_KEY:"[REDACTED]",DJANGO_SECRET_KEY:"[REDACTED]"});
  const input = 'DJANGO_SECRET_KEY=LEAKME\nAWS_SECRET_ACCESS_KEY=LEAKME\n-----BEGIN PRIVATE KEY-----\n'+'LEAKME'.repeat(100)+'\n-----END PRIVATE KEY-----\nnext line\n';
  assert.doesNotMatch(redact(input),/LEAKME/);
  for (const size of [1,7,256,4096]) {
    const redactor = new BinaryRedactor([]);
    const chunks: Buffer[] = [];
    for(let i=0;i<input.length;i+=size) chunks.push(redactor.push(Buffer.from(input.slice(i,i+size))));
    chunks.push(redactor.finish());
    const output = Buffer.concat(chunks).toString();
    assert.doesNotMatch(output,/LEAKME/);
    assert.match(output,/next line/);
  }
});

test("feedback after terminal cleanup keeps ordering without retaining per-run maps", async () => {
  const {AgentTelemetry} = await telemetryModule();
  const telemetry = Object.create(AgentTelemetry.prototype);
  const records: any[] = [];
  Object.assign(telemetry,{sequences:new Map(),versions:new Map(),runContext:new Map(),incomplete:new Map(),issuedSecrets:new Map(),secrets:[],send:async (message:any)=>records.push(...message.records)});
  await telemetry.record("r","s","run",{status:"completed"});
  telemetry.releaseRun("r");
  await telemetry.record("r","s","feedback",{verdict:"accepted"});
  await telemetry.record("r","s","feedback",{verdict:"needs_changes"});
  assert.ok(records[1].sequence > records[0].sequence);
  assert.ok(records[2].sequence > records[1].sequence);
  assert.equal(telemetry.sequences.size,0);
  assert.equal(telemetry.versions.size,0);
});
