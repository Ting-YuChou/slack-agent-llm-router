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
    JSON.stringify({ type: "tool_execution_start", toolCallId: "t1", toolName: "read", args: { path: "src/a.ts" } }),
    JSON.stringify({ type: "extension_ui_request", id: "approval-1", method: "confirm", title: "Allow edit?", message: "src/a.ts" }),
    JSON.stringify({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "Done" }, { type: "thinking", thinking: "hidden" }] } }),
    JSON.stringify({ type: "agent_settled" }),
  ].join("\n") + "\n"));
  bridge.respondToUi("approval-1", true);
  bridge.abort("abort-1");

  assert.deepEqual(emitted.map((event: any) => event.type), ["tool", "approval", "answer", "settled"]);
  assert.doesNotMatch(JSON.stringify(emitted), /hidden|thinking/);
  assert.deepEqual(JSON.parse(writes[0]), { type: "extension_ui_response", id: "approval-1", confirmed: true });
  assert.deepEqual(JSON.parse(writes[1]), { id: "abort-1", type: "abort" });
});
