import assert from "node:assert/strict";
import { test } from "node:test";

import { PiRpcBridge, RpcJsonlDecoder, RpcProtocolError, sanitizeRpcEvent } from "../src/pi-rpc.js";

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

test("RPC responses correlate by request id while public events interleave", async () => {
  const emitted: unknown[] = [];
  const writes: string[] = [];
  const bridge = new PiRpcBridge({
    writeLine: (line) => writes.push(line),
    onEvent: (event) => emitted.push(event),
  });

  const statePromise = bridge.getState(100);
  const statsPromise = bridge.getSessionStats(100);
  const stateRequest = JSON.parse(writes[0]);
  const statsRequest = JSON.parse(writes[1]);
  bridge.feed([
    JSON.stringify({ type: "compaction_start", reason: "threshold" }),
    JSON.stringify({
      id: statsRequest.id,
      type: "response",
      command: "get_session_stats",
      success: true,
      data: { sessionId: "pi-1", userMessages: 2, assistantMessages: 2, toolCalls: 1, toolResults: 1, totalMessages: 6, tokens: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, total: 15 }, cost: 0.01 },
    }),
    JSON.stringify({
      id: stateRequest.id,
      type: "response",
      command: "get_state",
      success: true,
      data: { sessionId: "pi-1", sessionFile: "/var/lib/pi-session/a.jsonl", autoCompactionEnabled: true, messageCount: 4, pendingMessageCount: 0, isStreaming: false, isCompacting: false, thinkingLevel: "max", steeringMode: "one-at-a-time", followUpMode: "one-at-a-time" },
    }),
    JSON.stringify({ type: "compaction_end", reason: "threshold", aborted: false, willRetry: false, result: { summary: "private summary" } }),
  ].join("\n") + "\n");

  assert.equal((await statePromise).sessionId, "pi-1");
  assert.equal((await statsPromise).toolCalls, 1);
  assert.deepEqual(emitted, [
    { type: "compaction", phase: "start", reason: "threshold" },
    { type: "compaction", phase: "end", reason: "threshold", aborted: false, will_retry: false },
  ]);
  assert.doesNotMatch(JSON.stringify(emitted), /private summary/);
});

test("RPC bridge rejects unknown response ids and times out pending commands", async () => {
  const bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent: () => undefined });
  assert.throws(
    () => bridge.feed(`${JSON.stringify({ id: "unknown", type: "response", command: "get_state", success: true, data: {} })}\n`),
    RpcProtocolError,
  );
  await assert.rejects(bridge.getTree(5), /timed out/i);
});

test("RPC bridge rejects malformed command-specific payloads", async () => {
  const writes: string[] = [];
  const bridge = new PiRpcBridge({
    writeLine: (line) => writes.push(line),
    onEvent: () => undefined,
  });
  const state = bridge.getState(100);
  const stateRequest = JSON.parse(writes[0]);
  bridge.feed(`${JSON.stringify({
    id: stateRequest.id,
    type: "response",
    command: "get_state",
    success: true,
    data: {
      sessionId: "pi-1",
      autoCompactionEnabled: "yes",
      messageCount: 1,
      pendingMessageCount: 0,
      isStreaming: false,
      isCompacting: false,
    },
  })}\n`);
  await assert.rejects(state, RpcProtocolError);

  const tree = bridge.getTree(100);
  const treeRequest = JSON.parse(writes[1]);
  bridge.feed(`${JSON.stringify({
    id: treeRequest.id,
    type: "response",
    command: "get_tree",
    success: true,
    data: {
      tree: [{ entry: { type: "message", id: "u1", parentId: null }, children: {} }],
      leafId: "u1",
    },
  })}\n`);
  await assert.rejects(tree, RpcProtocolError);
});
