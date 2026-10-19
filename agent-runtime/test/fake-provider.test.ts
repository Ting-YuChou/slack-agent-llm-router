import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { fakeGatewayScript, scriptedTurns, startFakeProvider } from "./support/fake-provider.js";

/** Parses an SSE body into its `event:` names (when present) and decoded `data:` payloads. */
function parseSse(body: string): Array<{ event?: string; data: unknown }> {
  return body.split("\n\n").filter(Boolean).map(block => {
    const lines = block.split("\n");
    const event = lines.find(line => line.startsWith("event: "))?.slice(7);
    const raw = lines.find(line => line.startsWith("data: "))!.slice(6);
    return { ...(event ? { event } : {}), data: raw === "[DONE]" ? "[DONE]" : JSON.parse(raw) };
  });
}
const types = (body: string) => parseSse(body).map(({ data }) => (data === "[DONE]" ? data : (data as { type: string }).type));
const post = (url: string, body: unknown, route = "/openai/v1/responses") =>
  fetch(`${url}${route}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

test("scriptedTurns builds the Responses SSE sequence for a function call", () => {
  const [turn] = scriptedTurns([{ kind: "function_call", name: "write", arguments: { path: "a.txt", content: "x\n" } }]);
  assert.equal(turn.protocol, "openai");
  assert.deepEqual(turn.events.map(event => event.type), [
    "response.created", "response.output_item.added", "response.function_call_arguments.done", "response.output_item.done", "response.completed",
  ]);
  const item = { type: "function_call", id: "fc_1", call_id: "call_1", name: "write", arguments: "{\"path\":\"a.txt\",\"content\":\"x\\n\"}", status: "completed" };
  assert.deepEqual(turn.events[1], { type: "response.output_item.added", output_index: 0, item });
  assert.deepEqual(turn.events[2], { type: "response.function_call_arguments.done", output_index: 0, arguments: item.arguments });
  assert.deepEqual(turn.events[4], {
    type: "response.completed",
    response: { id: "resp_1", status: "completed", output: [item], usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2, input_tokens_details: { cached_tokens: 0 } } },
  });
});

test("scriptedTurns numbers ids by turn and carries text, custom usage and string arguments", () => {
  const turns = scriptedTurns([
    { kind: "function_call", name: "run_checks", arguments: "{\"checks\":[\"lint\"]}" },
    { kind: "text", text: "done", usage: { input_tokens: 7, output_tokens: 3 } },
  ]);
  assert.equal((turns[0].events[1].item as { arguments: string }).arguments, "{\"checks\":[\"lint\"]}");
  assert.deepEqual(turns[1].events.map(event => event.type), [
    "response.created", "response.output_item.added", "response.output_text.delta", "response.output_item.done", "response.completed",
  ]);
  const item = { type: "message", id: "msg_2", role: "assistant", status: "completed", content: [{ type: "output_text", text: "done", annotations: [] }], phase: "final_answer" };
  assert.deepEqual(turns[1].events[2], { type: "response.output_text.delta", output_index: 0, content_index: 0, delta: "done" });
  assert.deepEqual(turns[1].events[4], {
    type: "response.completed",
    response: { id: "resp_2", status: "completed", output: [item], usage: { input_tokens: 7, output_tokens: 3, total_tokens: 10, input_tokens_details: { cached_tokens: 0 } } },
  });
});

test("scriptedTurns ends a max_output_tokens turn with response.incomplete", () => {
  const [turn] = scriptedTurns([{ kind: "incomplete", reason: "max_output_tokens", text: "partial" }]);
  const last = turn.events.at(-1) as { type: string; response: Record<string, unknown> };
  assert.equal(last.type, "response.incomplete");
  assert.equal(last.response.status, "incomplete");
  assert.deepEqual(last.response.incomplete_details, { reason: "max_output_tokens" });
  assert.equal((last.response.output as Array<{ status: string }>)[0].status, "incomplete");
  assert.ok(turn.events.some(event => event.type === "response.output_text.delta"));
});

test("fake provider answers /healthz without consuming a turn and serves turns in order", async () => {
  const provider = await startFakeProvider(scriptedTurns([
    { kind: "function_call", name: "write", arguments: { path: "a" } },
    { kind: "text", text: "second" },
  ]));
  try {
    const health = await fetch(`${provider.url}/healthz`);
    assert.equal(health.status, 200);
    assert.equal(await health.text(), "ok");
    assert.equal(provider.calls, 0);

    const first = await post(provider.url, { model: "m", input: "one" });
    assert.equal(first.status, 200);
    assert.equal(first.headers.get("content-type"), "text/event-stream");
    const body = await first.text();
    assert.deepEqual(types(body), [
      "response.created", "response.output_item.added", "response.function_call_arguments.done", "response.output_item.done", "response.completed", "[DONE]",
    ]);
    assert.equal(provider.calls, 1);

    const second = await post(provider.url, { model: "m", input: "two" });
    assert.deepEqual(types(await second.text()).at(-2), "response.completed");
    assert.equal(provider.calls, 2);
    assert.deepEqual(provider.requests.map(request => [request.method, request.url, (request.body as { input: string }).input]), [
      ["POST", "/openai/v1/responses", "one"], ["POST", "/openai/v1/responses", "two"],
    ]);

    const extra = await post(provider.url, { model: "m" });
    assert.equal(extra.status, 500);
    assert.match(await extra.text(), /no scripted turn 3/);
  } finally {
    await provider.close();
  }
});

test("fake provider rejects a request that misses the turn's expectations and keeps the turn", async () => {
  const provider = await startFakeProvider(scriptedTurns([
    { kind: "text", text: "ok", expect: { model: "gpt-5.6-luna", effort: "max", includes: ["needle"], excludes: ["mcp__evil__"] } },
  ]));
  try {
    const wrongModel = await post(provider.url, { model: "gpt-5.6-sol", reasoning: { effort: "max" }, input: "needle" });
    assert.equal(wrongModel.status, 400);
    assert.match(await wrongModel.text(), /model/);
    const wrongEffort = await post(provider.url, { model: "gpt-5.6-luna", reasoning: { effort: "high" }, input: "needle" });
    assert.equal(wrongEffort.status, 400);
    assert.match(await wrongEffort.text(), /effort/);
    const missing = await post(provider.url, { model: "gpt-5.6-luna", reasoning: { effort: "max" }, input: "hay" });
    assert.equal(missing.status, 400);
    assert.match(await missing.text(), /needle/);
    const forbidden = await post(provider.url, { model: "gpt-5.6-luna", reasoning: { effort: "max" }, input: "needle mcp__evil__x" });
    assert.equal(forbidden.status, 400);
    assert.match(await forbidden.text(), /mcp__evil__/);
    assert.equal(provider.calls, 0);

    const good = await post(provider.url, { model: "gpt-5.6-luna", reasoning: { effort: "max" }, input: "needle" });
    assert.equal(good.status, 200);
    assert.equal(provider.calls, 1);
  } finally {
    await provider.close();
  }
});

test("fake provider serves an Anthropic Messages turn with named SSE events", async () => {
  const provider = await startFakeProvider(scriptedTurns([{ kind: "anthropic_text", text: "anthropic-ok" }]));
  try {
    const wrongRoute = await post(provider.url, { model: "claude" });
    assert.equal(wrongRoute.status, 400);
    const response = await post(provider.url, { model: "claude" }, "/anthropic/v1/messages?beta=true");
    assert.equal(response.status, 200);
    const events = parseSse(await response.text());
    assert.deepEqual(events.map(({ event }) => event), ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
    assert.deepEqual(events[2].data, { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "anthropic-ok" } });
    assert.equal(provider.calls, 1);
  } finally {
    await provider.close();
  }
});

test("fakeGatewayScript runs standalone under node -e, answers /healthz and records requests", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "fake-provider-"));
  const recordPath = path.join(dir, "requests.jsonl");
  const child = spawn(process.execPath, ["-e", fakeGatewayScript(scriptedTurns([{ kind: "text", text: "hi" }]), { port: 0, recordPath })], { stdio: ["ignore", "pipe", "inherit"] });
  try {
    const port = await new Promise<number>((resolve, reject) => {
      let out = "";
      child.stdout.on("data", chunk => {
        out += chunk;
        const match = /listening on (\d+)/.exec(out);
        if (match) resolve(Number(match[1]));
      });
      child.once("exit", code => reject(new Error(`fake gateway exited ${code}`)));
    });
    const url = `http://127.0.0.1:${port}`;
    assert.equal((await fetch(`${url}/healthz`)).status, 200);
    const response = await post(url, { model: "m", input: "recorded" });
    assert.equal(types(await response.text()).at(-2), "response.completed");
    const records = (await readFile(recordPath, "utf8")).trim().split("\n").map(line => JSON.parse(line));
    assert.equal(records.length, 1);
    assert.equal(records[0].url, "/openai/v1/responses");
    assert.equal(records[0].body.input, "recorded");
    assert.equal(records[0].headers["content-type"], "application/json");
  } finally {
    child.kill();
    await rm(dir, { recursive: true, force: true });
  }
});
