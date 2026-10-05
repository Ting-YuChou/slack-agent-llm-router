import assert from "node:assert/strict";
import { test } from "node:test";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
async function tracingModule(): Promise<any> { const name = "../src/tracing.js"; const m = await import(name).catch(() => null); assert.ok(m, "OTEL tracing module exists"); return m; }
test("tools pair by call id, gateway parent is signed run context, unfinished tools are marked", async () => {
  const { AgentTracing } = await tracingModule();
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  tracing.startRun("r", "s");
  tracing.lifecycle("r", "routing", "start");
  tracing.lifecycle("r", "routing", "end");
  tracing.rpc("r", { type: "tool_execution_start", toolCallId: "a", toolName: "bash" });
  tracing.rpc("r", { type: "tool_execution_start", toolCallId: "b", toolName: "read" });
  tracing.rpc("r", { type: "tool_execution_end", toolCallId: "b", isError: false });
  const parent = tracing.traceparent("r");
  assert.match(parent, /^00-[a-f0-9]{32}-[a-f0-9]{16}-01$/);
  const gateway = tracing.gateway("r", parent, "openai", "model");
  gateway.end();
  tracing.endRun("r", "cancelled");
  await tracing.flush();
  const spans = exporter.getFinishedSpans();
  assert.ok(spans.every((s: any) => s.attributes['agent.run_id'] === 'r'), "every child span can be filtered by run id");
  assert.equal(spans.filter((s: any) => s.name === "agent.tool").length, 2);
  assert.equal(spans.find((s: any) => s.attributes['tool.call_id'] === 'a')!.attributes['capture.incomplete'], true);
  const root = spans.find((s: any) => s.name === "agent.run");
  assert.equal(spans.find((s: any) => s.name === "gen_ai.request")!.spanContext().traceId, root!.spanContext().traceId);
  await tracing.close();
});
import { createModelGateway, closeModelGateway } from "../src/model-gateway.js";
import { issueGatewayToken } from "../src/gateway-token.js";
test("gateway trace ends after the final streamed byte and never forwards trace headers upstream", async () => {
  const { AgentTracing } = await tracingModule();
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  tracing.startRun("r", "s");
  let finish!: () => void;
  const body = new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode("first")); finish = () => { controller.enqueue(new TextEncoder().encode("last")); controller.close(); }; } });
  const server = createModelGateway({
    signingSecret: "secret", providerApiKeys: { openai: "key" }, tracing,
    fetchFn: async (_url: any, init: any) => { assert.equal(init.headers.traceparent, undefined); return new Response(body, { status: 200, headers: { 'x-request-id': 'request-1' } }); }
  } as any);
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as any;
  const token = issueGatewayToken({ runId: "r", provider: "openai", model: "gpt-5.6-luna", api: "openai-responses", reasoningEffort: "max", expiresAt: Date.now() + 60000, traceparent: tracing.traceparent("r") }, "secret");
  try {
    const response = await fetch(`http://127.0.0.1:${address.port}/openai/v1/responses`, { method: "POST", headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' }, body: JSON.stringify({ model: "gpt-5.6-luna", reasoning: { effort: "max" }, stream: true }) });
    assert.equal(response.status, 200);
    await tracing.flush();
    assert.equal(exporter.getFinishedSpans().length, 0);
    finish();
    assert.equal(await response.text(), "firstlast");
    await tracing.flush();
    const spans = exporter.getFinishedSpans();
    assert.equal(spans.length, 1);
    assert.equal(spans[0].attributes['gen_ai.response.request_id'], 'request-1');
  }
  finally {
    try {
      finish();
    }
    catch { }
    await new Promise<void>(resolve => server.close(() => resolve()));
    tracing.endRun('r', 'completed');
    await tracing.close();
  }
});

test("client disconnect aborts the upstream stream and marks the gateway span failed", async () => {
  const { AgentTracing } = await tracingModule();
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  tracing.startRun("r", "s");
  let aborted!: () => void;
  const upstreamAborted = new Promise<void>(resolve => { aborted = resolve; });
  const server = createModelGateway({
    signingSecret: "secret", providerApiKeys: { openai: "key" }, tracing,
    fetchFn: async (_url: any, init: any) => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("first"));
        init.signal.addEventListener("abort", () => {
          controller.error(new Error("client disconnected"));
          aborted();
        }, { once: true });
      }
    }))
  } as any);
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as any;
  const token = issueGatewayToken({ runId: "r", provider: "openai", model: "gpt-5.6-luna", api: "openai-responses", reasoningEffort: "max", expiresAt: Date.now() + 60000, traceparent: tracing.traceparent("r") }, "secret");
  const disconnect = new AbortController();
  try {
    const response = await fetch(`http://127.0.0.1:${address.port}/openai/v1/responses`, {
      method: "POST", signal: disconnect.signal,
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-5.6-luna", reasoning: { effort: "max" }, stream: true })
    });
    await response.body!.getReader().read();
    disconnect.abort();
    await Promise.race([upstreamAborted, new Promise((_, reject) => {
      const timer = setTimeout(() => reject(new Error("upstream was not aborted")), 2000);
      timer.unref();
    })]);
    await tracing.flush();
    const span = exporter.getFinishedSpans()[0];
    assert.equal(span.attributes["http.client_disconnected"], true);
    assert.equal(span.status.code, 2);
    assert.equal(typeof span.attributes["http.first_stream_data_ms"], "number");
  } finally {
    disconnect.abort();
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    tracing.endRun("r", "cancelled");
    await tracing.close();
  }
});

test("gateway shutdown closes active streams before flushing telemetry", async () => {
  let closedStreams = false;
  let flushed = false;
  let closed!: () => void;
  const server = {
    close(callback: () => void) { closed = callback; },
    closeAllConnections() { closedStreams = true; closed(); }
  };
  const tracing = { async close() { assert.equal(closedStreams, true); flushed = true; } };
  await closeModelGateway(server as any, tracing as any, 5);
  assert.equal(flushed, true);
});
