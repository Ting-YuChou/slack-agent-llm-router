import assert from "node:assert/strict";
import { test } from "node:test";

import { PiRpcBridge, RpcJsonlDecoder, sanitizeRpcEvent } from "../src/pi-rpc.js";

test("strict JSONL decoder keeps unicode separators inside a JSON string", () => {
  const decoder = new RpcJsonlDecoder();
  const records = [
    ...decoder.push(Buffer.from('{"type":"message_update","text":"a\u2028b"}\r\n{"type":')),
    ...decoder.push(Buffer.from('"agent_settled"}\n')),
  ];

  assert.equal(records.length, 2);
  assert.equal(records[0].text, "a\u2028b");
  assert.equal(records[1].type, "agent_settled");
});

test("thinking deltas and thinking blocks never leave the RPC bridge", () => {
  assert.equal(
    sanitizeRpcEvent({
      type: "message_update",
      assistantMessageEvent: { type: "thinking_delta", delta: "secret chain" },
      message: { role: "assistant", content: [{ type: "thinking", thinking: "secret" }] },
    }),
    null,
  );
  const safe = sanitizeRpcEvent({
    type: "message_update",
    assistantMessageEvent: { type: "text_delta", delta: "hello" },
    message: { role: "assistant", content: [{ type: "thinking", thinking: "secret" }, { type: "text", text: "hello" }] },
  });
  assert.doesNotMatch(JSON.stringify(safe), /secret|thinking/);
});

test("bridge maps tool, approval, final answer, and settled events", () => {
  const emitted: unknown[] = [];
  const writes: string[] = [];
  const bridge = new PiRpcBridge({
    writeLine: (line) => writes.push(line),
    onEvent: (event) => emitted.push(event),
  });

  bridge.feed(Buffer.from([
    JSON.stringify({ type: "turn_start" }),
    JSON.stringify({ type: "tool_execution_start", toolCallId: "t1", toolName: "read", args: { path: "src/a.ts" } }),
    JSON.stringify({ type: "extension_ui_request", id: "approval-1", method: "confirm", title: "Allow edit?", message: "src/a.ts" }),
    JSON.stringify({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "Done" }, { type: "thinking", thinking: "hidden" }] } }),
    JSON.stringify({ type: "agent_settled" }),
  ].join("\n") + "\n"));
  bridge.respondToUi("approval-1", true);
  bridge.abort("abort-1");

  assert.deepEqual(emitted.map((event: any) => event.type), ["turn", "tool", "approval", "answer", "settled"]);
  assert.doesNotMatch(JSON.stringify(emitted), /hidden|thinking/);
  assert.deepEqual(JSON.parse(writes[0]), { type: "extension_ui_response", id: "approval-1", confirmed: true });
  assert.deepEqual(JSON.parse(writes[1]), { id: "abort-1", type: "abort" });
});

test("bridge sets model and thinking level before sending a prompt", async () => {
  const writes: Array<Record<string, any>> = [];
  const bridge = new PiRpcBridge({
    writeLine: (line) => {
      const command = JSON.parse(line);
      writes.push(command);
      if (command.type === "set_model" || command.type === "set_thinking_level") {
        queueMicrotask(() => bridge.feed(JSON.stringify({ id: command.id, type: "response", command: command.type, success: true }) + "\n"));
      }
      if (command.type === "get_state") {
        queueMicrotask(() => bridge.feed(JSON.stringify({ id: command.id, type: "response", command: "get_state", success: true,
          data: { model: { provider: "openai", id: "gpt-5.6-sol" }, thinkingLevel: "high" } }) + "\n"));
      }
    },
    onEvent: () => undefined,
  });
  await (bridge as any).configureAndPrompt("run-1", "fix", "openai", "gpt-5.6-sol", "high");
  assert.deepEqual(writes.map((item) => item.type), ["set_model", "set_thinking_level", "get_state", "prompt"]);
  assert.equal(writes[3].message, "fix");
});

test("bridge refuses to prompt when restored Pi state does not match the gateway token", async () => {
  const writes: Array<Record<string, any>> = [];
  const bridge = new PiRpcBridge({
    writeLine: (line) => {
      const command = JSON.parse(line);
      writes.push(command);
      queueMicrotask(() => bridge.feed(JSON.stringify({ id: command.id, type: "response", command: command.type, success: true,
        data: command.type === "get_state" ? { model: { provider: "openai", id: "gpt-5.6-luna" }, thinkingLevel: "max" } : undefined }) + "\n"));
    },
    onEvent: () => undefined,
  });
  await assert.rejects((bridge as any).configureAndPrompt("run-1", "fix", "openai", "gpt-5.6-sol", "high"), /state mismatch/i);
  assert.ok(writes.every((item) => item.type !== "prompt"));
});

test("bridge forwards billed assistant cost without exposing response content", () => {
  const events: unknown[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent: (event) => events.push(event) });
  bridge.feed(JSON.stringify({ type: "message_end", message: { role: "assistant", content: [], usage: { cost: { total: 0.004 } } } }) + "\n");
  assert.deepEqual(events, [{ type: "usage", cost_usd: 0.004 }]);
});

test("MCP tool events retain only names, status and result byte count", () => {
  const start = sanitizeRpcEvent({
    type: "tool_execution_start", toolCallId: "m1", toolName: "mcp__github__search_code",
    args: { query: "repo:acme/widgets secret search", authorization: "Bearer secret" },
  });
  const end = sanitizeRpcEvent({
    type: "tool_execution_end", toolCallId: "m1", toolName: "mcp__github__search_code",
    result: { content: [{ type: "text", text: "private result" }] }, isError: false,
  });
  assert.equal(JSON.stringify(start).includes("secret"), false);
  assert.equal((end as any).result_bytes > 0, true);
  assert.equal(JSON.stringify(end).includes("private result"), false);
});

function settledAfter(records: Array<Record<string, unknown>>, bridge?: PiRpcBridge) {
  const events: any[] = [];
  const target = bridge ?? new PiRpcBridge({ writeLine: () => undefined, onEvent: (event) => events.push(event) });
  if (!bridge) target.prompt("run-1", "fix");
  target.feed(records.map((record) => JSON.stringify(record)).join("\n") + "\n");
  return events.filter((event) => event.type === "settled");
}

const assistantEnd = (stopReason: string, extra: Record<string, unknown> = {}) =>
  ({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "x" }], stopReason, ...extra } });

test("settled reports the last assistant stop reason as the run outcome", () => {
  assert.deepEqual(settledAfter([assistantEnd("stop"), { type: "agent_settled" }]).map((e) => e.outcome), ["completed"]);
  assert.deepEqual(settledAfter([assistantEnd("length"), { type: "agent_settled" }]).map((e) => e.outcome), ["truncated"]);
  assert.deepEqual(settledAfter([assistantEnd("aborted"), { type: "agent_settled" }]).map((e) => e.outcome), ["aborted"]);
});

test("settled error outcome carries a bounded, redacted provider message", () => {
  const [settled] = settledAfter([assistantEnd("error", { errorMessage: `upstream failed Bearer sk-secret-token ${"z".repeat(900)}` }), { type: "agent_settled" }]);
  assert.equal(settled.outcome, "error");
  assert.equal(settled.stop_reason, "error");
  assert.ok(settled.error_message.length <= 500);
  assert.doesNotMatch(settled.error_message, /sk-secret-token/);
});

test("auto-retry: a later successful assistant message wins over an earlier error", () => {
  const outcomes = settledAfter([assistantEnd("error", { errorMessage: "429" }), assistantEnd("stop"), { type: "agent_settled" }]).map((e) => e.outcome);
  assert.deepEqual(outcomes, ["completed"]);
});

test("tool-use stops are intermediate; the final assistant message decides", () => {
  assert.deepEqual(settledAfter([assistantEnd("toolUse"), assistantEnd("stop"), { type: "agent_settled" }]).map((e) => e.outcome), ["completed"]);
});

test("a policy limit entry before an abort reports which limit was hit", () => {
  const limit = { type: "entry_appended", entry: { type: "custom", customType: "pi_agent_limit", data: { kind: "tool_calls", limit: 40 } } };
  const [settled] = settledAfter([limit, assistantEnd("aborted"), { type: "agent_settled" }]);
  assert.equal(settled.outcome, "limit_reached");
  assert.equal(settled.limit, "tool_calls");
});

test("settling without any assistant message is an error, not a completion", () => {
  const [settled] = settledAfter([{ type: "agent_settled" }]);
  assert.equal(settled.outcome, "error");
  assert.equal(settled.stop_reason, "no_response");
});

test("outcome tracking resets for each prompt", () => {
  const events: any[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent: (event) => events.push(event) });
  bridge.prompt("run-1", "first");
  bridge.feed([assistantEnd("length"), { type: "agent_settled" }].map((r) => JSON.stringify(r)).join("\n") + "\n");
  bridge.prompt("run-2", "second");
  bridge.feed(JSON.stringify({ type: "agent_settled" }) + "\n");
  assert.deepEqual(events.filter((e) => e.type === "settled").map((e) => e.outcome), ["truncated", "error"]);
});

test("a rejected prompt command surfaces as an error instead of waiting for the deadline", () => {
  const events: any[] = [];
  const bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent: (event) => events.push(event) });
  bridge.prompt("run-1", "fix");
  bridge.feed(JSON.stringify({ id: "run-1", type: "response", command: "prompt", success: false, error: "busy" }) + "\n");
  assert.deepEqual(events, [{ type: "error", code: "prompt_rejected", message: "Pi rejected the prompt before it started" }]);
});
